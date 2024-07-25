
import numpy as np
import torch
import control
import utils
import monotonicnetworks as lmn

class PolicyModel(torch.nn.Module):

    def __init__(self, state_dim, action_dim,  model_config={}):
        super(PolicyModel, self).__init__()

        hidden_dim = model_config.get('hidden_dim', 100)
        self.hidden_dim = hidden_dim

        self.linear1 = torch.nn.Linear(state_dim, 2 * hidden_dim)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_dim, action_dim)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x

    def compute_loss(self, data):
        x = data['observations']
        u = data['actions']
        next_x = data['next_observations']
        c = data['costs']
        r = data['rewards']

        # === calculate loss ===
        u_pred = self.forward(x)
        action_loss = torch.nn.functional.mse_loss(u_pred, u)
        return action_loss, {'action_loss': action_loss}



class LatentLinearModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, model_config={}):
        super(LatentLinearModel, self).__init__()

        # === parameters ====
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = model_config.get('hidden_dim', 100)
        self.target_state = model_config.get('target_state', None)
        # MIMO for companion
        self.dynamic_structure = model_config.get('dynamic_structure', 'diag')
        if self.dynamic_structure == 'companion':
            self.hidden_dim = self.hidden_dim * action_dim
        self.c_latent_state = 1.0
        self.c_cost = model_config.get('c_cost', 1.0)
        self.c_next_state = model_config.get('c_next_state', 0.0)
        self.c_action = model_config.get('c_action', 0.0)
        self.stop_grad = model_config.get('stop_grad', False)

        # === embed layer ====
        self.state_emb = torch.nn.Sequential(torch.nn.Linear(state_dim, 2 * self.hidden_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim))

        # === linear transition ====
        if self.dynamic_structure == 'subdiag':
            r = self.hidden_dim // action_dim - 1
            A = torch.zeros((self.hidden_dim, self.hidden_dim))
            A[range(0, self.hidden_dim-action_dim), range(action_dim, self.hidden_dim)] = 1.0
            self.A = torch.nn.Parameter(A, requires_grad=False)
            B = torch.zeros((self.hidden_dim, action_dim))
            B[-action_dim:, :] = 1.0
            self.B = torch.nn.Parameter(B, requires_grad=False)
        elif self.dynamic_structure == 'companion':
            # TODO: use randn to init
            self.A_raw = torch.nn.Parameter(torch.zeros(action_dim, self.hidden_dim), requires_grad=True) # requires_grad = False, torch.randn
            self.B_raw = torch.nn.Parameter(torch.zeros(action_dim*(action_dim-1)//2), requires_grad=True) # requires_grad = False, torch.randn
        elif self.dynamic_structure == 'diag':
            self.A_raw = torch.nn.Parameter(torch.randn((self.hidden_dim)))
            self.B = torch.nn.Parameter(torch.randn((self.hidden_dim, action_dim)))
        else:
            self.A = torch.nn.Parameter(torch.randn((self.hidden_dim, self.hidden_dim)))
            self.B = torch.nn.Parameter(torch.randn((self.hidden_dim, action_dim)))

        # === cost functions ===
        self.cost_structure = model_config.get('cost_structure',  'companion')
        if 'psd' in self.cost_structure:
            # TODO: use zeros to init
            self.Q_raw = torch.nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        elif 'diag' in self.cost_structure:
            self.Q_raw = torch.nn.Parameter(torch.randn(self.hidden_dim))
        else:
            self.Q = torch.nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim)) # without psd constraint
        # M: fix R now
        self.R = torch.nn.Parameter(0.1*torch.eye(action_dim), requires_grad=False)

        if 'monotonic' in self.cost_structure:
            width = self.hidden_dim
            self.cost_head = torch.nn.Sequential(
                    lmn.direct_norm(torch.nn.Linear(1, width), kind="one-inf"),
                    lmn.GroupSort(2),
                    lmn.direct_norm(torch.nn.Linear(width, width), kind="inf"),
                    lmn.GroupSort(2),
                    lmn.direct_norm(torch.nn.Linear(width, 1), kind="inf"))
        self.K = None

        # === task heads ===
        # M: no need to use
        self.action_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(2 * self.hidden_dim, action_dim))

        self.next_state_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(2 * self.hidden_dim, state_dim))
        # TODO: gradients/value predictions
        # self.value_head = None
        # self.gradient_head = None

        # === optimizer ===
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)


    @property
    def device(self):
        return next(self.parameters()).device

    def get_lqr_matrics(self, to_numpy=True):

        if self.dynamic_structure == 'diag':
            self.A = torch.zeros_like(self.Q_raw)
            self.A[range(self.hidden_dim), range(self.hidden_dim)] = self.A_raw
        elif self.dynamic_structure == 'companion':
            hidden_dim, action_dim = self.hidden_dim, self.action_dim
            raw_dim = hidden_dim // action_dim
            self.A = torch.zeros((hidden_dim, hidden_dim), device=self.A_raw.device)
            self.B = torch.zeros((self.hidden_dim, action_dim), device=self.B_raw.device)
            # self.B[range(raw_dim-1, self.hidden_dim, raw_dim), range(0, action_dim)] = 1.0
            b_id = 0
            for i in range(action_dim):
                self.A[range(i*raw_dim, (i+1)*raw_dim-1), range(i*raw_dim+1, (i+1)*raw_dim)] = 1.0
                self.A[(i+1)*raw_dim-1, :] = self.A_raw[i]
                # self.A[(i+1)*raw_dim-1, range(i*raw_dim, (i+1)*raw_dim)] = self.A_raw[i, range(i*raw_dim, (i+1)*raw_dim)]
                self.B[(i+1)*raw_dim-1, i] = 1.0
                self.B[(i+1)*raw_dim-1, i+1:] = self.B_raw[b_id:b_id+action_dim-i-1]
                b_id += action_dim-i-1

        if 'psd' in self.cost_structure:
            Q = torch.matmul(self.Q_raw, self.Q_raw.t())
            self.Q = Q + torch.eye(self.hidden_dim).to(Q.device)
            # R = torch.matmul(self.R_raw, self.R_raw.t())
            # self.R = R + torch.eye(self.action_dim).to(R.device)
        elif 'diag' in self.cost_structure:
            self.Q = torch.zeros((self.hidden_dim, self.hidden_dim), device=self.A.device)
            self.Q[range(self.hidden_dim), range(self.hidden_dim)] = self.Q_raw * self.Q_raw
            # self.R = torch.zeros((self.action_dim, self.action_dim), device=self.A.device)
            # self.R[range(self.action_dim), range(self.action_dim)] = self.R_raw

        if to_numpy:
            A = self.A.data.cpu().numpy()
            B = self.B.data.cpu().numpy()
            Q = self.Q.data.cpu().numpy()
            R = self.R.data.cpu().numpy()
            return A, B, Q, R
        else:
            return self.A, self.B, self.Q, self.R

    def forward(self, x):
        # Predict in latent space
        z = self.state_emb(x)
        if self.K is None:
            A,B,Q,R = self.get_lqr_matrics(to_numpy=True)
            K, S, E = control.dlqr(A, B, Q, R)
            self.K = utils.np_to_torch(K, device=z.device)
        u = - torch.matmul(z, torch.transpose(self.K, 1, 0))
        return u

    def compute_loss(self, data):

        x = data['observations']
        u = data['actions']
        next_x = data['next_observations']
        c = data['costs']
        r = data['rewards']

        A, B, Q, R = self.get_lqr_matrics(to_numpy=False)

        # === calculate loss ===
        z = self.state_emb(x)
        next_z = self.state_emb(next_x)

        # consistency loss: z' = Az + Bu
        next_z_pred = torch.transpose(torch.matmul(A, torch.transpose(z, 1, 0)) + torch.matmul(B, torch.transpose(u, 1, 0)), 1, 0)
        if self.stop_grad:
            next_z = next_z.detach()
        latent_state_loss = torch.nn.functional.mse_loss(next_z, next_z_pred)

        # next state loss
        next_x_pred = self.next_state_head(next_z_pred)
        next_state_loss = torch.nn.functional.mse_loss(next_x_pred, next_x)

        # cost loss
        # target_error = next_z_pred
        target_error = z

        q_loss = torch.diag(torch.matmul(torch.matmul(target_error, Q), torch.transpose(target_error, 1, 0)))
        r_loss = 0
        # r_loss = torch.diag(torch.matmul(torch.matmul(u, R), torch.transpose(u,1,0)))
        raw_cost_loss = torch.unsqueeze(q_loss + r_loss, dim=-1)
        if 'monotonic' in self.cost_structure:
            raw_cost_loss = self.cost_head(raw_cost_loss)
        cost_loss = torch.nn.functional.mse_loss(raw_cost_loss[:,0], -r)

        # if self.target_state:
        #     target_state = torch.unsqueeze(torch.tensor(self.target_state, device=next_z_pred.device), dim=0)
        #     target_error -= self.state_emb(target_state)
        # if self.target_state:
        #     target_state = torch.unsqueeze(torch.tensor(self.target_state, device=next_z_pred.device), dim=0)
        #     target_state = self.state_emb(target_state)
        #     target_loss = torch.nn.functional.mse_loss(target_state, torch.zeros_like(target_state))
        #     cost_loss += 0.1 * target_loss

        # action loss
        u_pred = self.action_head(z)
        action_loss = torch.nn.functional.mse_loss(u_pred, u)

        # total loss
        total_loss = self.c_latent_state * latent_state_loss + self.c_next_state * next_state_loss + self.c_cost * cost_loss + self.c_action * action_loss

        loss_dict = {'latent_state_loss': latent_state_loss.item(), 'next_state_loss': next_state_loss.item(), 'cost_loss': cost_loss.item(), 'action_loss': action_loss.item()}

        try:
            # check control properties
            # - controlability
            A,B,Q,R = self.get_lqr_matrics()
            C = control.ctrb(A, B)
            rank_c = np.linalg.matrix_rank(C)
            loss_dict['rank_c'] = rank_c

            # - eigenvalues
            K, S, E = control.dlqr(A, B, Q, R)
            I = A - np.matmul(B,K)
            loss_dict['eigen_i_norm'] = np.trace(np.matmul(I, np.transpose(I)))
            evalues, evectors = np.linalg.eig(I)
            eigen_i = np.abs(evalues)
            loss_dict['eigen_i_max'] = np.max(eigen_i)
            loss_dict['eigen_i_min'] = np.min(eigen_i)

            # - eigenloss: only for cartpole
            state_zero = torch.tensor(np.zeros((1, 4), dtype=np.float32), device=x.device, requires_grad=True) # [1, 4]
            jacob_phi = torch.autograd.functional.jacobian(self.state_emb, state_zero, create_graph=True)[0, :, 0, :] # [20, 4]
            evalues = torch.tensor([0.99402952+0.00445417j, 0.99402952-0.00445417j, 0.99753357+0.00132452j, 0.99753357-0.00132452j], device=x.device, requires_grad=False)
            evectors = torch.tensor([[ 6.75356892e-02+0.02753158j,  6.75356892e-02-0.02753158j, 2.13206327e-01+0.11486084j,  2.13206327e-01-0.11486084j], [-8.81535194e-02-0.06638229j, -8.81535194e-02+0.06638229j, 1.10293521e-01-0.20455585j,  1.10293521e-01+0.20455585j],  [-5.28383265e-01+0.1396258j,  -5.28383265e-01-0.1396258j, -6.79671343e-01+0.j,         -6.79671343e-01-0.j        ], [ 8.26933523e-01+0.j,          8.26933523e-01-0.j, -2.29680293e-04+0.6522186j,  -2.29680293e-04-0.6522186j ]], device=x.device, requires_grad=False)
            new_evectors = torch.matmul(jacob_phi.to(torch.complex64), evectors) # [20, 4]
            new_evalues = torch.diag(evalues) # [4, 4]
            I_th = torch.tensor(I, device=x.device, requires_grad=False).to(torch.complex64) # [20, 20]
            eigen_loss = torch.sum(torch.abs(torch.matmul(I_th, new_evectors) - torch.matmul(new_evectors, new_evalues)))
            loss_dict['eigen_loss'] = eigen_loss.item()
            # total_loss += 0.01 * eigen_loss

        except:
            loss_dict['eigen_i_norm'] = 0.0
            loss_dict['eigen_i_max'] = 0.0
            loss_dict['eigen_i_min'] = 0.0
            loss_dict['eigen_loss'] = 0.0


        return total_loss, loss_dict


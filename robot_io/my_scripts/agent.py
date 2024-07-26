
import numpy as np
import torch
import control


class LLMController(object):
    
    def __init__(self, model):
        self.model = model
        ms = model.get_lqr_matrics()
        A,B,Q,R = ms[:4]
        self.A, self.B, self.Q, self.R = A,B,Q,R
        K, S, E = control.dlqr(A, B, Q, R)
        self.K = K
        self.K_torch = torch.tensor(K.astype('float32'), device=model.device)

        # check properties
        # 1. controllability
        C = control.ctrb(A, B)
        print(f'Controlability: rank(C) {np.linalg.matrix_rank(C)}')
        
        # 2. stability
        I = A - np.matmul(B,K)
        evalues, evectors = np.linalg.eig(I)
        print(f'Stability: absolute of eigenvalues A-BK {np.abs(evalues)}')
        self.step = 0
        
 
    def act(self, state, step=0):
        state = torch.tensor(state.astype('float32'), device=self.model.device)
        state = torch.unsqueeze(state, dim=0)
        state_error = self.model.state_emb(state)[0] # z 
        action = - torch.matmul(self.K_torch, state_error).cpu().detach().numpy()
        return action
    

class ILController(object):
    
    def __init__(self, model):
        self.model = model
        
    def act(self, state, step=0):
        state = torch.tensor(state.astype('float32'), device=self.model.device)
        state = torch.unsqueeze(state, dim=0)
        action = self.model(state).cpu().detach().numpy()[0]
        return action



def get_frame(path, i):
    filename = os.path.join(path, f"frame_{i:06d}.npz")
    return np.load(filename, allow_pickle=True)


class TrajController(object):

    def __init__(self, path, (start_end_ids)):
        self.start_id, self.end_id = start_end_ids
        self.frames = [get_frame(path, i) for i in range(self.start_id, self.end_id+1)]
        self.init_robot_state = self.frames[0]['robot_state'].item()

    def act(self, state, step=0): 
        current_frame = self.frames[step] 
        action = frame["action"].item()
        return action





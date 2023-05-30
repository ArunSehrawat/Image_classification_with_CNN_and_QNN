import torch
import math



def get_device(gpu_no):
    if torch.cuda.is_available():
        return torch.device('cuda', gpu_no)
    else:
        return torch.device('cpu')


    


    
class quantum_circuit:
    
    def __init__(self, num_qubits : int, state_vector = None, device = 'cuda', gpu_no = 0):
        
        """
        Defines a quantum circuit object that stores the full state-vector (evolved through 
        the unitary operations of a quantum circuit) of `num_qubits` number of qubits. 

        Args:
            num_qubits (int): Number of qubits in the circuit.
            
            state_vector (torch.Tensor, optional): The full state vector of the quantum circuit. 
                                          Defaults to None. If None is provided then the state 
                                          vector is automatically initialized to the ket |0000...0>.
                                          
            device (str, optional): Device on which the state vector should be stored (CPU / GPU). 
                                    Defaults to 'cuda' i.e. GPU.
                                    
            gpu_no (int, optional): If there are multiple GPUs then this parameter defines which 
                                    GPU to use. Defaults to 0 i.e. the first device.
        """
        #---------------------------------------------------------------------------------------- 
        
        if device != 'cuda': 
            self.device = torch.device(device)
        else: 
            self.device = get_device(gpu_no)
            
        
        #----------------------------------------------------------------------------------------    
            

        self.n = num_qubits   # number of qubits
        self.dim = 2**self.n  # dimention of the n-qubit hilbert space
       
    
        #----------------------------------------------------------------------------------------  
        
        '''
        state_vector can 
        (1) either be a vector of shape (dim,)
        (2) either be a matrix of shape (dim, number of examples)
        '''
        
        if state_vector is None:
            ''' Initialize the state-vector to |0000...0> '''
            state_vector = torch.zeros(self.dim, device=self.device, dtype=torch.cfloat) 
            state_vector[0] = 1
            self.state_vector = state_vector.reshape(-1,1)
        else:
            if state_vector.shape[0] == self.dim: 
                ''' state_vector must be normalized '''
                self.state_vector = state_vector.to(torch.cfloat)
            else:
                print('The dimension 2**n does NOT match the shape of the state vector. n is the number of qubits.')
                

        #---------------------------------------------------------------------------------------- 
        
        # single qubit Pauli gates (matrices) :    
        self.I        = torch.tensor([[1,   0], [0,  1]], device=self.device, dtype=torch.cfloat)
        self.x_matrix = torch.tensor([[0.,  1], [1,  0]], device=self.device, dtype=torch.cfloat)
        self.y_matrix = torch.tensor([[0, -1j], [1j, 0]], device=self.device, dtype=torch.cfloat)
        self.z_matrix = torch.tensor([[1,   0], [0, -1]], device=self.device, dtype=torch.cfloat)

        self.h_matrix = (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], device=self.device, dtype=torch.cfloat)
        

        # single qubit projectors :
        self.proj_0 = torch.tensor([[1, 0], [0, 0]], device=self.device, dtype=torch.cfloat)
        self.proj_1 = torch.tensor([[0, 0], [0, 1]], device=self.device, dtype=torch.cfloat)
        
        
        
   #======================================================================================================

        
        
    def single_qubit_gate(self, target : int, gate : torch.Tensor):
        """
        Applies a single qubit gate = I ⊗ I ⊗ ... ⊗ gate ⊗ ... ⊗ I

        Args:
        target (int): The qubit index on which the gate will be applied
        gate (torch.Tensor): The matrix representation of a single qubit gate 

        Returns:
        The state vector of the full quantum circuit after applying the single qubit gate.

        """
        
        if target < 0 or self.n <= target: 
            print('0 <= traget <= num_qubits - 1 is NOT satisfied!')
            
        else:
            single_q_gate = torch.tensor(1, device=self.device, dtype=torch.cfloat) # initialize
            
            for k in range(self.n):
                if k == target:
                    single_q_gate = torch.kron(single_q_gate, gate)
                else:
                    single_q_gate = torch.kron(single_q_gate, self.I)
                   
            #------------------------------------------------------
            
            self.state_vector = torch.matmul(single_q_gate, self.state_vector) 
            return self.state_vector

    
    
    
    def controlled_gate(self, control: int, target: int, gate : torch.Tensor):
        """
        Applies a two-qubit controlled gate between the 'control` and `target` qubits.
        
        control_gate_part_0 = I ⊗ |0><0| ⊗ ... ⊗ I    ⊗ ... ⊗ I    
        control_gate_part_1 = I ⊗ |1><1| ⊗ ... ⊗ gate ⊗ ... ⊗ I         SEE: the control is set to 1
        
        control_gate = control_gate_part_0 + control_gate_part_1
        

        Args:
        control (int): Control qubit index
        target (int):  Target qubit index 
        gate (torch.Tensor): The matrix representation of a single qubit gate

        Returns:
        The state vector of the full quantum circuit after applying the two-qubit gate.
        """
        
        if control < 0 or self.n <= control: 
            print('0 <= control <= num_qubits - 1 is NOT satisfied!')   
        elif target < 0 or self.n <= target: 
            print('0 <= target <= num_qubits - 1 is NOT satisfied!') 
        elif control == target:
            print('control and traget qubits must be different!')
        else:
            control_gate_part_0 = torch.tensor(1, device=self.device, dtype=torch.cfloat) # initialize
            control_gate_part_1 = torch.tensor(1, device=self.device, dtype=torch.cfloat) 
            
            for k in range(self.n):
                if k == control:
                    control_gate_part_0 = torch.kron(control_gate_part_0, self.proj_0)
                    control_gate_part_1 = torch.kron(control_gate_part_1, self.proj_1)
                elif k == target:
                    control_gate_part_0 = torch.kron(control_gate_part_0, self.I)
                    control_gate_part_1 = torch.kron(control_gate_part_1, gate)
                else:
                    control_gate_part_0 = torch.kron(control_gate_part_0, self.I)
                    control_gate_part_1 = torch.kron(control_gate_part_1, self.I)
            
            control_gate = control_gate_part_0 + control_gate_part_1
            
            self.state_vector = torch.matmul(control_gate, self.state_vector)
            return self.state_vector
    

    #====================================================================================================== 
    
    def x(self, target : int):                           # Applies X gate (matrix) on the target qubit
        'NOTE: 0 <= target <= num_qubits - 1'
        self.single_qubit_gate(target, self.x_matrix)

        
    def y(self, target : int):
        self.single_qubit_gate(target, self.y_matrix)

        
    def z(self, target : int):
        self.single_qubit_gate(target, self.z_matrix)

        
    def h(self, target : int):                           # Applies Hadamard gate (matrix) on the target qubit
        self.single_qubit_gate(target, self.h_matrix)
   

   #======================================================================================================


    def Rx(self, target : int, theta):
        
        """
        Applies Rx gate (rotation around x axis) on the target qubit

        Args:
        theta (torch.Tensor): Angle by which the qubit should be rotated around X axis. 
                              Usually a tunable parameter is passed.
        
        target (int): Qubit index on which the Rx gate will be applied. 
        NOTE: 0 <= target <= num_qubits - 1
        """
        
        co = torch.cos(theta / 2)
        si = torch.sin(theta / 2)
        self.Rx_matrix = torch.stack([torch.stack([co, -1j*si]), torch.stack([-1j*si, co])])
        
        self.single_qubit_gate(target, self.Rx_matrix)
        
        
        
        
    def Ry(self, target : int, theta):            #like Rx, Ry gate applies (rotation around y axis) on the target qubit

        co = torch.cos(theta / 2)
        si = torch.sin(theta / 2)
        self.Ry_matrix = torch.stack([torch.stack([co, -si]), torch.stack([si, co])])
        
        self.single_qubit_gate(target, self.Ry_matrix)

      
    
    
        
    def Rz(self, target : int, theta):            #like Rx, Ry gate applies (rotation around z axis) on the target qubit

        exp_theta = torch.exp( 1j*theta )
        zero = torch.tensor(0)
        one = torch.tensor(1)
        self.Rz_matrix = torch.stack([torch.stack([one, zero]), torch.stack([zero, exp_theta])])      

        self.single_qubit_gate(target, self.Rz_matrix)
        
        
        
        
    def R(self, target : int, theta, phi, lamda):
        """
        Applies general rotation to the target qubit

        Args:
        theta, phi and lamda (torch.Tensor): The Euler angles which define a general rotation around Bloch sphere. 

        target (int): Qubit index on which the gate will be applied.
        """
        
        a =                                  torch.cos(theta / 2)
        b =        - torch.exp(1j * lamda) * torch.sin(theta / 2)
        c =            torch.exp(1j * phi) * torch.sin(theta / 2)
        d =  torch.exp(1j * (phi + lamda)) * torch.cos(theta / 2)
        self.R_matrix = torch.stack([torch.stack([a, b]), torch.stack([c, d])])
        
        self.single_qubit_gate(target, self.R_matrix)
        

    #====================================================================================================== 
    
    
    def Ry_layer(self, angs: torch.Tensor):
        '''
        Applies tensor-product of single-qubit rotations around y-axis
        '''
        
        cos, sin = torch.cos(angs[0]), torch.sin(angs[0])
        '''
        Use torch.stack otherwise computation graph will be broken (or will not begin).
        And, grad will be gone (will not be stored). 
        '''
        rot = torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])
        
        for i in range(1, len(angs)):                                    # one angles for each qubit         
            cos, sin = torch.cos(angs[i]), torch.sin(angs[i])
            rot = torch.kron(rot, torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])) 
                                                  
        #--------------------------------------------------------------------------
        
        self.state_vector = torch.matmul(rot, self.state_vector)      # rotated state vector
        return self.state_vector     
    
    
    
    
    
    def Rz_layer(self, angs: torch.Tensor):       #like Ry_layer, Rz_layer acts
        
        exp_ang = torch.exp( 1j*angs[0] )
        zero = torch.tensor(0)
        one = torch.tensor(1)
 
        rot = torch.stack([torch.stack([one, zero]), torch.stack([zero, exp_ang])]) 
        
        for i in range(1, len(angs)):                                    # one angles for each qubit         
            exp_ang = torch.exp( 1j*angs[i] )
            rot = torch.kron(rot, torch.stack([torch.stack([one, zero]), torch.stack([zero, exp_ang])]) ) 
                                                  
        #--------------------------------------------------------------------------
        
        self.state_vector = torch.matmul(rot, self.state_vector)      # rotated state vector
        return self.state_vector   
    
    
    
    #====================================================================================================== 
        
    def cx(self, control: int, target: int):
        """
        Applies controlled-X gate = I ⊗ |0><0| ⊗ ... ⊗ I ⊗ ... ⊗ I +
                                    I ⊗ |1><1| ⊗ ... ⊗ X ⊗ ... ⊗ I
                                    
        Args:
        control (int): Control qubit index
        target (int):  Target qubit index 
        """
        self.controlled_gate(control, target, self.x_matrix)

       
    
        
    def cz(self, control: int, target: int):                           #like cx, cz gate acts
        self.controlled_gate(control, target, self.z_matrix)

        
        
   #====================================================================================================== 


    def cx_linear_layer(self):
        '''
        Applies cx(n-1,n) ... cx(2,3) cx(1,2) cx(0,1) |state_vector>
        
        NOTE: First cx(0,1) will act on |state_vector>, then cx(1,2)
              And in the last cx(n-1,n) will act.
              order matter in case of cx
        '''
        
        self.controlled_gate(self.n - 2, self.n - 1, self.x_matrix)
        for i in range(self.n - 3, -1, -1):
            self.controlled_gate(i, i+1, self.x_matrix)
    
    
                                 
    def cz_linear_layer(self):                                        #like cx_linear_layer, cz_linear_layer  acts
        self.controlled_gate(self.n - 2, self.n - 1, self.z_matrix)
        for i in range(self.n - 3, -1, -1):
            self.controlled_gate(i, i+1, self.z_matrix)

    
   #====================================================================================================== 

    def probabilities(self):  
        """
        probabilities obtained in the z-measurement (computational basis) on the state vector

        Returns: A torch.Tensor of the size same as the state vector
        """
        
        return self.state_vector.conj() * self.state_vector

    
    

    
    
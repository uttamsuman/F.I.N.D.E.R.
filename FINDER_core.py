import torch
import numpy as np

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for FINDER")

class FINDER(torch.optim.Optimizer):
    """ base class for FINDER optimizer """
    
    def __init__(self,model = None, p = 5, R = None, Γ = None, Δ = None, θ = 0.9, γ = 1, cs = 0.1, clamps = [0.01, 0.01]):
        assert isinstance(p, int) and p >= 3, "Variable p must be a natural number greater than or equal to 3"
        if model != None: # used for ANN problems
            self.model = model
            self.state_X = torch.nn.utils.parameters_to_vector(self.model.parameters()).view(-1,1)
            self.N = self.state_X.shape[0]
            if self.model.loss_grad == None:
                raise AttributeError("Model object has no method named loss_grad.")
            else:
                self.loss_grad = self.model.loss_grad
        else:
            raise "No model passed while creating the object of class FINDER."
        defaults = dict(qwerty = "FINDER", θ = θ, γ = γ, cs = cs, clamps = clamps)
        super(FINDER, self).__init__(params=self.model.parameters(), defaults = defaults)
        self.θ = θ
        self.p = p
        self.γ = γ
        self.cs = cs
        self.clamps = clamps 
        if Δ == None:
            self.Δ = torch.zeros(self.N,self.p).to(device)
        else:
            self.Δ = Δ
        if Γ == None:
            self.Γ = torch.zeros(self.N,1).to(device)
        else:
            self.Γ = Γ
        if R == None:
            self.R = 0.1 * torch.ones(1,self.N).to(device)
        else:
            self.R = R
        
        self.y_grad = torch.zeros_like(self.Δ) # to store gradients
        self.xmin = self.state_X # to store best particle of ensemble
        self.xmax = torch.ones_like(self.xmin) # to store worst particle
        self.xmiin = torch.empty_like(self.state_X) # to store best of arx
        self.new_x = torch.empty_like(self.state_X) # next point for Armijo rule
        # in B computation
        self.a = torch.empty_like(self.Δ) 
        self.b = torch.empty_like(self.Δ)
        self.c = torch.empty_like(self.state_X)
        self.d = torch.empty_like(self.state_X)
        # to store ensemble
        self.arx = torch.empty_like(self.Δ)
        self.arx_new = torch.empty_like(self.Δ)
        self.new_ensemble = torch.empty(self.N, self.p+1).to(device)
        self.new_ensemble.requires_grad = False
        # to store B, B_
        self.B = torch.empty_like(self.state_X)
        self.B_ = torch.empty_like(self.state_X)
        # to store indices for sorting
        self.idx = torch.tensor([0])
        self.sorted_indices = torch.tensor([0]*self.p).to(device)
        # to store loss
        self.y = torch.empty(self.p+1).to(device)
        self.y0 = torch.tensor([0.0]).to(device)
        # to sample from uniform distribution
        self.rand = torch.empty(self.p-1, self.N).to(device)
        # to store increment of current iteration
        self.Hg = torch.empty_like(self.Δ)
        
    def generate_arx(self):
        """generates an ensemble of p particles by sampling (p-1) particles 
           with uniform distribution U(-1,1) along each component of state_X 
           and concatenating x to the sampled particles"""
        
        self.arx[:,:] = self.xmin
        self.arx[:,1:] = self.xmin + (self.R * self.rand.uniform_(-1,1)).T

    def x_y_grad(self, inputs, labels = None, no_grad = False):
        """set grads = True for loss and gradient computation on initial ensemble
           set grads = False for only loss computation on new ensemble"""
        if no_grad == False:
            for i, element in enumerate(self.arx.T):
                self.y[i], self.y_grad[:,i] = self.loss_grad(element, inputs, labels, no_grad)
            
            self.sorted_indices[:] = torch.argsort(self.y[:self.p])
            self.arx[:,:] = self.arx[:,self.sorted_indices]
            self.y_grad[:,:] = self.y_grad[:, self.sorted_indices]
            self.idx[:] = self.sorted_indices[0]
            self.y0[:] = self.y[self.idx]
            self.xmiin[:,0] = self.arx[:,0]
        
        else:
            for i, element in enumerate(self.new_ensemble.T):
                self.y[i] = self.loss_grad(element, inputs, labels, no_grad)
            self.y0[:] = self.y.min()
            
            self.xmin[:,0] = self.new_ensemble[:,torch.argmin(self.y)]
            self.xmax[:,0] = self.new_ensemble[:,torch.argmax(self.y)]
            

    def calc_invH(self):
        """directly computes B which is the diagonal approximation to inverse Hessian and returns B**γ"""
        with torch.no_grad():
            torch.sub(self.arx, torch.mean(self.arx, 1, True), out=self.a)
            torch.sub(self.y_grad, torch.mean(self.y_grad, 1, True), out=self.b)
            torch.sum(self.a*self.b, dim=1, out=self.c[:,0])
            torch.sum(self.b*self.b, dim=1, out=self.d[:,0])
            torch.div(self.c, self.d, out=self.B)
            torch.clamp(torch.nan_to_num(self.B), 0, out=self.B_)
            self.B_ **= self.γ


    def cal_sampling_radius(self):
        """ returns the sampling radius sigma and the evolution path variable ps for the next iteration"""
        with torch.no_grad():
            self.Γ[:] = (1 - self.cs) * self.Γ + self.cs * (self.xmax - self.xmin)
            self.R[0,:] = torch.min(torch.abs(self.Γ.T), self.clamps[0] * torch.ones_like(self.Γ.T))
            self.R = torch.where(self.R == 0, self.clamps[1], self.R)


    def update_ensemble(self):
        """returns an updated ensemble with the previous increment term for the next iteration"""
        self.arx_new[:,:] = self.arx
        self.arx_new[:,:] -= self.α * self.Δ
        

    def calc_increment(self):
        """returns the increment"""
        self.Δ *= self.θ
        torch.mul(self.B_, self.y_grad, out=self.Hg)
        self.Δ += self.Hg
        


    def Armijo_rule(self, inputs, labels = None):
        """Armijo rule for computation of step size multiplier"""
        step = 0.1
        c_α = 0.01
        pvec = - self.Δ[:,self.idx] # direction of descent
        δ = 1.0
        pvec1 = pvec.view(-1,1)
        g = self.y_grad[:,self.idx]
        
        while δ > 1e-6:
            self.new_x[:] = self.xmiin - δ * self.Δ[:,self.idx:self.idx+1]
            loss_new = self.loss_grad(self.new_x, inputs, labels, no_grad=True)
            min_loss = - c_α * δ * torch.dot(self.Δ[:,self.idx.item()], self.y_grad[:,self.idx.item()])
            armijo = loss_new - self.y0 - min_loss.item()
            if armijo < -0:
                step = δ
                break
            else:
                δ *= 0.5
        return step


    def step(self, inputs, labels = None):
        
        '''generate inital ensemble say X'''
        self.generate_arx()
        
        '''loss and gradient computation and finding fittest particle in initial ensemble'''
        self.x_y_grad(inputs, labels, no_grad=False)
        
        '''compute B_ = B**γ'''
        self.calc_invH()
        
        '''compute increment term Δ by convolution of past increments'''
        self.calc_increment()
        
        '''compute step size multiplier α using Armijo rule'''
        self.α = self.Armijo_rule(inputs, labels)

        '''update the initial ensemble'''
        self.update_ensemble()

        '''concatenate fittest particle from initial ensemble to updated ensemble'''
        self.new_ensemble[:,0:1] = self.xmiin
        self.new_ensemble[:,1:] = self.arx_new

        '''loss computation without gradient computation'''
        self.x_y_grad(inputs, labels, no_grad=True)

        '''compute parameter-wise sampling radius'''
        self.cal_sampling_radius()
        
        self.state_X[:] = self.xmin
        
        torch.nn.utils.vector_to_parameters(self.xmin, self.model.parameters())

        return self.y0.item()


import torch

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for FINDER")

class FINDER(torch.optim.Optimizer):
    """ base class for FINDER optimizer """
    
    def __init__(self,model = None, p = 5, R = None, Γ = None, Δ = None):
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
        defaults = dict(qwerty = "FINDER", θ = 0.9, γ = 1, cs = 0.1, clamps = [0.01, 0.01])
        super(FINDER, self).__init__(params=self.model.parameters(), defaults = defaults)

        self.p = p
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
        
        self.y_grad = torch.zeros_like(self.Δ)
        self.xmin = self.state_X
        self.xmax = torch.ones_like(self.xmin)
        self.a = torch.empty_like(self.Δ)
        self.b = torch.empty_like(self.Δ)
        self.c = torch.empty_like(self.state_X)
        self.d = torch.empty_like(self.state_X)
        self.arx = torch.empty_like(self.Δ)
        self.arx_new = torch.empty_like(self.Δ)
        self.new_ensemble = torch.empty(self.N, self.p+1).to(device)
        self.B = torch.empty_like(self.state_X)
        self.B_ = torch.empty_like(self.state_X)
        self.xmiin = torch.empty_like(self.state_X)
        self.new_x = torch.empty_like(self.state_X)
        self.idx = 0
        
    def generate_arx(self):
        """generates an ensemble of p particles by sampling (p-1) particles 
           with uniform distribution U(-1,1) along each component of state_X 
           and concatenating x to the sampled particles"""
        self.arx[:,:] = torch.cat([self.xmin, self.xmin + (self.R * (2*torch.rand(self.p-1, self.N).to(device)-1)).T], dim = 1)

    # @record_function("sorted_x_y_grad")
    def sorted_x_y_grad(self, inputs, labels = None, no_grad = False):
        """set grads = True for loss and gradient computation on initial ensemble
           set grads = False for only loss computation on new ensemble"""
        y = []
        if no_grad == False:
            for i, element in enumerate(self.arx.T):
                loss, self.y_grad[:,i] = self.loss_grad(element, inputs, labels, no_grad)
                y.append(loss.item())
            
            self.y0 = min(y)
            self.idx = y.index(self.y0)
            self.xmiin[:,0] = self.arx[:,self.idx]
            
        
        else:
            for element in self.new_ensemble.T:
                y.append(self.loss_grad(element, inputs, labels, no_grad).item())
            self.y0 = min(y)
            
            self.xmin[:,0] = self.new_ensemble[:,y.index(self.y0)]
            self.xmax[:,0] = self.new_ensemble[:,y.index(max(y))]
            

    def calc_invH(self, γ = 1):
        """directly computes B which is the diagonal approximation to inverse Hessian and returns B**γ"""
        self.a[:,:] = self.arx - torch.mean(self.arx, 1, True)
        self.b[:,:] = self.y_grad - torch.mean(self.y_grad, 1, True)
        self.c[:,0] = torch.sum(self.a*self.b, dim = 1)
        self.d[:,0] = torch.sum(self.b*self.b, dim = 1)
        self.B[:] = (self.c / self.d)
        self.B_[:] = torch.clamp(torch.nan_to_num(self.B), 0)**γ


    def cal_sampling_radius(self, clamps = [0.01, 0.01], cs = 0.1):
        """ returns the sampling radius sigma and the evolution path variable ps for the next iteration"""
        self.Γ[:] = (1 - cs) * self.Γ + cs * (self.xmax - self.xmin)
        self.R[0,:] = torch.min(torch.abs(self.Γ.T), clamps[0] * torch.ones_like(self.Γ.T))
        self.R[self.R == 0] = clamps[1]


    def update_ensemble(self):
        """returns an updated ensemble with the previous increment term for the next iteration"""
        self.arx_new[:,:] = self.arx - self.α * self.Δ
        

    def calc_increment(self, θ=0.9):
        """returns the increment"""
        self.Δ[:,:] = θ * self.Δ + self.B_ * self.y_grad


    def Armijo_rule(self, inputs, labels = None):
        """Armijo rule for computation of step size multiplier"""
        step = 0.1
        c_1 = 0.01
        pvec = - self.Δ[:,self.idx] #Hg # direction of descent
        δ = 1.0
        pvec1 = pvec.view(-1,1)
        g = self.y_grad[:,self.idx]
        
        while δ > 1e-6:
            self.new_x[:] = self.xmiin - δ * self.Δ[:,self.idx:self.idx+1]
            loss_new = self.loss_grad(self.new_x, inputs, labels, no_grad=True).item()
            min_loss = - c_1 * δ * self.Δ[:,self.idx] @ self.y_grad[:,self.idx]
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
        self.sorted_x_y_grad(inputs, labels, no_grad=False)
        
        '''compute B_ = B**γ'''
        self.calc_invH()
        
        '''compute increment term Δ by convolution of past increments'''
        self.calc_increment()
        
        '''compute step size multiplier α using Armijo rule'''
        self.α = self.Armijo_rule(inputs, labels)

        '''update the initial ensemble'''
        self.update_ensemble()

        '''concatenate fittest particle from initial ensemble to updated ensemble'''
        self.new_ensemble[:,:] = torch.cat([self.arx_new, self.xmiin], dim = 1)

        '''loss computation without gradient computation'''
        self.sorted_x_y_grad(inputs, labels, no_grad=True)

        '''compute parameter-wise sampling radius'''
        self.cal_sampling_radius()
        
        self.state_X[:] = self.xmin
        
        torch.nn.utils.vector_to_parameters(self.xmin, self.model.parameters())

        return self.y0
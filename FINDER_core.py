import torch
from memory_profiler import profile

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for FINDER")

class FINDER(torch.optim.Optimizer):
    """ base class for FINDER optimizer """
    
    def __init__(self,model = None):
        
        if model != None: # used for ANN problems
            self.model = model
            self.state_X = torch.nn.utils.parameters_to_vector(self.model.parameters()).reshape(-1,1)
            self.N = self.state_X.shape[0]
            if self.model.loss_grad == None:
                raise AttributeError("module object has no method named loss_grad")
            else:
                self.loss_grad = self.model.loss_grad

        defaults = dict(qwerty = "FINDER")
        super(FINDER, self).__init__(params=self.model.parameters(), defaults = defaults)

    def generate_arx(self,x, r, p = 5):
        """generates an ensemble of p particles by sampling (p-1) particles 
           with uniform distribution U(-1,1) along each component of state_X 
           and concatenating x to the sampled particles"""
        
        return torch.cat([x, x + (r.reshape(1,-1) * (2*torch.rand(p-1, self.N).to(device)-1)).T], dim = 1)

    
    def sorted_x_y_grad(self, x, inputs, labels = None, no_grad = False):
        """set grads = True for loss and gradient computation on initial ensemble
           set grads = False for only loss computation on new ensemble"""
        
        y = []
        y_grad = []
        if no_grad == False:
            for element in x.T:
                loss1, gradients1 = self.loss_grad(element, inputs, labels, no_grad)
                y.append(loss1)
                y_grad.append(gradients1)
                
            y_grad = torch.hstack(y_grad)
            y0 = min(y)
            idx = y.index(y0)
            fittest_x = x[:,idx].reshape(-1,1)
            return y0, y_grad, fittest_x, idx
        
        else:
            for element in x.T:
                loss1 = self.loss_grad(element, inputs, labels, no_grad)
                y.append(loss1)
            y0 = min(y)
            
            fittest = x[:,y.index(y0)].view(-1,1)
            worst = x[:,y.index(max(y))].view(-1,1)

            return fittest, worst, y0


    def calc_invH(self, x, x_grad, γ = 1):
        """directly computes B which is the diagonal approximation to inverse Hessian and returns B**γ"""
        a = x - torch.mean(x, 1, True)
        b = x_grad - torch.mean(x_grad, 1, True)
        c = torch.sum(a*b, dim = 1)
        d = torch.sum(b*b, dim = 1)
        Hinv = (c / d).reshape(-1,1)
        Hinv = torch.clamp(torch.nan_to_num(Hinv), 0)
        return Hinv**γ


    def cal_sampling_radius(self, xmin, xmax, ps, clamps = [0.01, 0.01], cs = 0.1):
        """ returns the sampling radius sigma and the evolution path variable ps for the next iteration"""
        ps = (1 - cs) * ps + cs * (xmax - xmin)
        sig = torch.min(torch.abs(ps.T), clamps[0] * torch.ones_like(ps.T))
        sig[sig == 0] = clamps[1]
        return sig, ps


    def update_ensemble(self, x, α, Δ):
        """returns an updated ensemble with the previous increment term for the next iteration"""
        x1 = x - α * Δ
        return x1


    def increment(self, Δ, B, G, θ=0.9):
        """returns the increment"""
        return θ * Δ + B * G


    def Armijo_rule(self, g, Hg, x, loss, inputs, labels = None):
        """Armijo rule for computation of step size multiplier"""
        step = 0.1
        c_1 = 0.01
        pvec = - Hg # direction of descent
        δ = 1.0
        pvec1 = pvec.reshape(-1,1)
        while δ > 1e-6:
            new_x = x + δ * pvec1
            loss_new = self.loss_grad(new_x, inputs, labels, no_grad=True)
            min_loss = c_1 * δ * pvec @ g
            armijo = loss_new - loss - min_loss
            if armijo < -0:
                step = δ
                break
            else:
                δ *= 0.5
        return step


    def step(self, inputs, labels = None, xmin = None, R = None, Γ = None, Δ = None, p = 5):
        
        '''initializations'''
        if xmin == None:
            xmin = self.state_X
        if R == None:
            R = 0.1 * torch.ones(1,self.N).to(device)
        if Γ == None:
            Γ = torch.zeros(self.N,1).to(device)
        if Δ == None:
            Δ = torch.zeros(self.N, p).to(device)
        
        '''generate inital ensemble say X'''
        arx = self.generate_arx(xmin, R, p)
        
        '''loss and gradient computation and finding fittest particle in initial ensemble'''
        least_fitness_value, gradf, xmiin, idx = self.sorted_x_y_grad(arx, inputs, labels = None, no_grad=False)
        
        '''compute B_ = B**γ'''
        B_ = self.calc_invH(arx, gradf)
        
        '''compute increment term Δ by convolution of past increments'''
        Δ = self.increment(Δ, B_, gradf)
        
        '''compute step size multiplier α using Armijo rule'''
        α = self.Armijo_rule( gradf[:, idx], Δ[:,idx], xmiin, least_fitness_value, inputs, labels)

        '''update the initial ensemble'''
        arx_new = self.update_ensemble(arx, α, Δ)

        '''concatenate fittest particle from initial ensemble to updated ensemble'''
        new_ensemble = torch.cat([arx_new, xmiin], dim = 1)

        '''loss computation without gradient computation'''
        xmin, xmax, min_fitness_value = self.sorted_x_y_grad(new_ensemble, inputs, labels = None, no_grad=True)

        '''compute parameter-wise sampling radius'''
        R, Γ = self.cal_sampling_radius(xmin, xmax, Γ)
        
        self.state_X = xmin
        
        torch.nn.utils.vector_to_parameters(xmin, self.model.parameters())

        return xmin, R, Γ, Δ, min_fitness_value
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

#df.parameters['form_compiler']['quadrature_degree'] = 4

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)
def Min(a, b): return (a+b-abs(a-b))/df.Constant(2)

# HELPER FUNCTIONS
def softplus(y1,y2,alpha=1):
    # The softplus function is a differentiable approximation
    # to the ramp function.  Its derivative is the logistic function.
    # Larger alpha makes a sharper transition.
    return Max(y1,y2) + alpha*df.ln(1+df.exp(1./alpha*(Min(y1,y2)-Max(y1,y2))))

# Sub domain for Periodic boundary condition
class PeriodicBoundary(df.SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < df.DOLFIN_EPS and x[0] > -df.DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

# Create periodic boundary condition
pbc = PeriodicBoundary()

mesh = df.Mesh('square.xml')

a = df.Constant(0.1)

E_cg = df.FiniteElement('CG',mesh.ufl_cell(),1)
E_dg = df.FiniteElement('DG',mesh.ufl_cell(),0)

E_V = df.MixedElement([E_cg,E_cg,E_dg,E_cg])

Q_cg = df.FunctionSpace(mesh,E_cg)
Q_dg = df.FunctionSpace(mesh,E_dg)
V = df.FunctionSpace(mesh,E_V)

class H_ex(df.UserExpression):
    def eval(self,values,x):
        r = np.sqrt((x[1]-0.5)**2)
        values[0] = 1e-2*max(1-16*r**2,1e-2)
class B_ex(df.UserExpression):
    def eval(self,values,x):
        values[0] = (-0.2*np.exp(-(x[0]-0.5)**2/0.1**2))

class r0_ex(df.UserExpression):
    def eval(self,values,x):
        r = np.sqrt((x[1]-0.5)**2)
        values[0] = (r - (np.sqrt(1./16)))

class adot(df.UserExpression):
    def eval(self,values,x): 
        r = np.sqrt((x[1]-0.5)**2)
        values[0] = 100*(1-16*r**2) - 20

H0 = df.interpolate(H_ex(),Q_dg)
Bhat = df.interpolate(B_ex(),Q_cg)
phi0 = df.interpolate(r0_ex(),Q_cg)

u0 = df.Function(Q_cg)
u0.vector()[:] = 1
v0 = df.Function(Q_cg)
v0.vector()[:] = 1

U = df.Function(V)
Phi = df.TestFunction(V)
dU = df.TrialFunction(V)

u,v,H,phi = df.split(U)
lamda_x,lamda_y,xsi,w = df.split(Phi)

# normal vector
nhat = df.FacetNormal(mesh)

theta = df.TestFunction(Q_cg)
du = df.TrialFunction(Q_cg)

# Constants
n = 3
b = 1e-16**(-1./n)
beta2 = 100
rho = 917
rho_w = 1000
g = 9.81
L = 1e5
Z = 1000

# Geometric variables
d = Max(-Bhat,1/Z)                 # Water depth
l = Max(0,Bhat+1/Z)                # water surface height
B = Max(Bhat,-rho/rho_w*H)         # Ice base
N = H - Max(rho_w/rho*(l-B),0.8*H) # Effective pressure

dt = df.Constant(1e-8)

# Non-dimensional groups
U_ = (rho*g*Z)/(beta2*L)
gamma = b/2*(U_/L)**(1./n)/(rho*g*Z)
tau_ = L/U_

# Signum and Heaviside function
S = phi/df.sqrt(phi**2 + df.CellDiameter(mesh)**2)
Heav = (S + 1)*0.5

# Effective pressure regularization
N0 = 1e-3

# SSA stress balance
eta = (u.dx(0)**2 + v.dx(1)**2 + u.dx(0)*v.dx(1) + 0.25*(u.dx(1) + v.dx(0))**2 + (L/U_)**2*1e-10)**((1-n)/(2*n))
R_u = (-gamma*lamda_x.dx(0)*2*eta*H*(2*u.dx(0) + v.dx(1)) - gamma*lamda_x.dx(1)*eta*H*(u.dx(1) + v.dx(0)) - lamda_x*(N+N0)*u - lamda_x*H*B.dx(0) + 0.5*H**2*lamda_x.dx(0))*df.dx - 0.5*H**2*lamda_x*nhat[0]*df.ds
R_v = (-gamma*lamda_y.dx(0)*eta*H*(u.dx(1) + v.dx(0)) - gamma*lamda_y.dx(1)*2*eta*H*(u.dx(0) + 2*v.dx(1)) - lamda_y*(N+N0)*v - lamda_y*H*B.dx(1) + 0.5*H**2*lamda_y.dx(1))*df.dx - 0.5*H**2*lamda_y*nhat[1]*df.ds

# Crank Nicholson parameter
cn = 1.0
Hm = cn*H + (1-cn)*H0

# Jump and average for DG elements
H_avg = 0.5*(Hm('+') + Hm('-'))
H_jump = Hm('+')*nhat('+') + Hm('-')*nhat('-')

xsi_avg = 0.5*(xsi('+') + xsi('-'))
xsi_jump = xsi('+')*nhat('+') + xsi('-')*nhat('-')

uvec = df.as_vector([u,v])
unorm = df.dot(uvec,uvec)**0.5

#Upwind
uH = df.avg(uvec)*H_avg + 0.5*df.avg(unorm)*H_jump
beff = df.Function(Q_dg)
beff.interpolate(adot())
R_H = ((H-H0)/dt - beff)*xsi*df.dx - df.dot(df.grad(xsi),uvec*Hm)*df.dx + df.dot(uH,xsi_jump)*df.dS# + xsi*H*df.dot(uvec,nhat)*df.ds

# Calving relations
k_calving = df.Constant(0.5)
H_calving = (1+df.Constant(.0))*d*rho_w/rho
f = Min((H_calving/H),5)
ucalv = f*uvec

# ALE level set velocity
unorm_c = df.dot(uvec-ucalv,uvec-ucalv)**0.5

# Artificial diffusion for level set transport
k_a = df.CellDiameter(mesh)/2.*unorm_c

R_xsi = (phi-phi0)/dt*w*df.dx - df.dot(df.grad(w),(uvec-ucalv)*phi)*df.dx + w*df.dot((uvec-ucalv),df.grad(phi))*df.dx + df.dot(df.grad(w),k_a*df.grad(phi))*df.dx + w*phi*df.dot((uvec-ucalv),nhat)*df.ds

# Coupled form for stress balance, mass balance, and level set
R = R_u + R_v + R_H + R_xsi
J = df.derivative(R,U,dU)

# Forms necessary for transforming the level set function back into a signed distance function (i.e. the reinitialization equations)
phif = df.Function(Q_cg)
phif_0 = df.Function(Q_cg)
tau = dt*df.Constant(50)

k_aa = df.Constant(5e-3)

normal = -df.grad(phif_0)/df.sqrt(df.dot(df.grad(phif_0),df.grad(phif_0))+1e-5)
R_fic = ((du - phif_0)/tau*theta - S*theta*(1-df.sqrt(df.dot(df.grad(phif_0),df.grad(phif_0)))))*df.dx + k_aa*df.dot(df.grad(theta),normal)*df.dot(df.grad(du),normal)*df.dx

# Define forms
a_form = df.lhs(R_fic)
L_form = df.rhs(R_fic)

# paraview files
Hfile = df.File('demo/H.pvd')
phifile = df.File('demo/phi.pvd')
ufile = df.File('demo/u.pvd')
vfile = df.File('demo/v.pvd')

# Function assigners (maps from the coupled function U to the individual components)
assigner_inv = df.FunctionAssigner([Q_cg,Q_cg,Q_dg,Q_cg],V)
assigner = df.FunctionAssigner(V,[Q_cg,Q_cg,Q_dg,Q_cg])

# Set initial guess for U to be its initial values
assigner.assign(U,[u0,v0,H0,phi0])

# positivity constraint on thickness
l_bound = df.Function(V)
u_bound = df.Function(V)
u_bound.vector()[:] = 1e10
lc = df.Function(Q_cg)
lc.vector()[:] = -1e10
thklim = 0.1/Z
ld = df.Function(Q_dg)
ld.vector()[:] = thklim
assigner.assign(l_bound,[lc,lc,ld,lc])

# Nonlinear Problem
problem = df.NonlinearVariationalProblem(R,U,J=J)
problem.set_bounds(l_bound,u_bound)

n_smoothing_iterations = 50

t = 0
t_end = 2000

while t<t_end:
    try:
        assigner.assign(U,[u0,v0,H0,phi0])

        # Reinitialize solver inside loop (weird petsc bug)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters['nonlinear_solver'] = 'snes'
        solver.parameters['snes_solver']['method'] = 'vinewtonssls'
        solver.parameters['snes_solver']['relative_tolerance'] = 1e-3
        solver.parameters['snes_solver']['absolute_tolerance'] = 1e-3
        solver.parameters['snes_solver']['error_on_nonconvergence'] = True
        solver.parameters['snes_solver']['linear_solver'] = 'mumps'
        solver.parameters['snes_solver']['maximum_iterations'] = 10
        solver.parameters['snes_solver']['report'] = True

        # Solve coupled system
        solver.solve()
        assigner_inv.assign([u0,v0,H0,phi0],U)
        Hfile << (H0,t*tau_)
        phifile << (phi0,t*tau_)
        ufile << (u0,t*tau_)
        vfile << (v0,t*tau_)
        
        # Reinitialize level set
        phif_0.vector()[:] = phi0.vector()[:]
        for k in range(n_smoothing_iterations):
            df.solve(a_form==L_form,phif,solver_parameters={'linear_solver':'lu'},form_compiler_parameters={'optimize':True})
            phif_0.vector()[:] = phif.vector()[:]
        
        phi0.vector()[:] = phif.vector()[:]
        print('(time step),(system mass): ',dt(0)*tau_,df.assemble(H*df.dx)) 

        t+=dt(0)
        dt.assign(min(dt(0)*1.1,20/tau_))
    except RuntimeError:
        dt.assign(dt(0)/2.)
        print('solver failed, reducing time step and trying again:',dt(0))


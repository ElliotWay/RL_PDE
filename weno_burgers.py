import numpy
from matplotlib import pyplot
import burgers
import weno_coefficients
from scipy.optimize import brentq
import sys
import time

def burgers_sine_exact(x, t):
    """
    Compute the exact solution of Burger's given the 'sine' initial data
    """
#    def initial_sin(x):
#        if x < 1/3 or x > 2/3:
#            return 1
#        else:
#            return 1 + numpy.sin(6*numpy.pi*(x-1/3))/2
    def initial_smooth_sin(x):
        return numpy.sin(2*numpy.pi*x)
    def initial_gaussian(x):
        return 1.0 + numpy.exp(-60.0*(x - 0.5)**2)
    def residual(x_at_t_0_guess, x_at_t):
        q = initial_gaussian(x_at_t_0_guess)
        return x_at_t_0_guess + q * t - x_at_t
    
    q = numpy.zeros_like(x)
    for i in range(len(q)):
        x_at_t_0 = brentq(residual, -2, 2, args=(x[i],))
        q[i] = initial_gaussian(x_at_t_0)
    return q
        

def weno(order, q):
    """
    Do WENO reconstruction
    
    Parameters
    ----------
    
    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct
        
    Returns
    -------
    
    qL : numpy array
        Reconstructed data - boundary points are zero
    """
    C = weno_coefficients.C_all[order]
    a = weno_coefficients.a_all[order]
    sigma = weno_coefficients.sigma_all[order]

    qL = numpy.zeros_like(q)
    beta = numpy.zeros((order, len(q)))
    w = numpy.zeros_like(beta)
    np = len(q) - 2 * order
    epsilon = 1e-16
    for i in range(order, np+order):
        q_stencils = numpy.zeros(order)
        alpha = numpy.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l+1):
                    beta[k, i] += sigma[k, l, m] * q[i+k-l] * q[i+k-m]
            alpha[k] = C[k] / (epsilon + beta[k, i]**2)
            for l in range(order):
                q_stencils[k] += a[k, l] * q[i+k-l]
        w[:, i] = alpha / numpy.sum(alpha)
        qL[i] = numpy.dot(w[:, i], q_stencils)
    
    return qL

def weno_i(order, q):
    """
    Do WENO reconstruction at a given location
    
    Parameters
    ----------
    
    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct
        
    Returns
    -------
    
    qL : float
        Reconstructed data - boundary points are zero
    """
    C = weno_coefficients.C_all[order]
    a = weno_coefficients.a_all[order]
    sigma = weno_coefficients.sigma_all[order]

    qL = numpy.zeros_like(q)
    beta = numpy.zeros((order))
    w = numpy.zeros_like(beta)
    epsilon = 1e-16
    q_stencils = numpy.zeros(order)
    alpha = numpy.zeros(order)
    for k in range(order):
        for l in range(order):
            for m in range(l+1):
                beta[k] += sigma[k, l, m] * q[order-1+k-l] * q[order-1+k-m]
        alpha[k] = C[k] / (epsilon + beta[k]**2)
        for l in range(order):
            q_stencils[k] += a[k, l] * q[order-1+k-l]
    w[:] = alpha / numpy.sum(alpha)
    qL = numpy.dot(w[:], q_stencils)

    return qL
  
def weno_i_stencils(order, q):
  """
  Get stencils at a given location in the grid

  Parameters
  ----------
  order : int
    The stencil width.
  q : numpy array
    flux vector for that stencil.

  Returns
  -------
  None.

  """
  a = weno_coefficients.a_all[order]
  q_stencils = numpy.zeros(order)
  #TODO: remove these loops; generate matrix for q and do matmul
  for k in range(order):
    for l in range(order):
      q_stencils[k] += a[k, l] * q[order-1+k-l]
      
  return q_stencils

def weno_i_stencils_batch(order, q_batch):
  """
  Take a batch of pieces of state and returns the stencil values.

  Parameters
  ----------
  order : int
    DESCRIPTION.
  q_batch : numpy array
    DESCRIPTION.

  Returns
  -------
  Return a batch of stencils

  """
  
  q_fp_stencil = []
  q_fm_stencil = []
  batch_size = q_batch.shape[1]
  for i in range(batch_size):
    q_fp_stencil.append(weno_i_stencils(order, q_batch[0,i,:]))
    q_fm_stencil.append(weno_i_stencils(order, q_batch[1,i,:]))
    
  return numpy.array([q_fp_stencil, q_fm_stencil])

def weno_i_weights(order, q):
  """
  Get WENO weights at a given location in the grid

  Parameters
  ----------
  order : int
    The stencil width.
  q : numpy array
    flux vector for that stencil.

  Returns
  -------
  None.

  """
  C = weno_coefficients.C_all[order]
  sigma = weno_coefficients.sigma_all[order]

  beta = numpy.zeros((order))
  w = numpy.zeros_like(beta)
  epsilon = 1e-16
  alpha = numpy.zeros(order)
  for k in range(order):
      for l in range(order):
          for m in range(l+1):
              beta[k] += sigma[k, l, m] * q[order-1+k-l] * q[order-1+k-m]
      alpha[k] = C[k] / (epsilon + beta[k]**2)
  w[:] = alpha / numpy.sum(alpha)
  
  return w

def weno_i_weights_batch(order, q_batch):
  """
  Get WENO weights for a batch

  Parameters
  ----------
  order : TYPE
    DESCRIPTION.
  q_batch : TYPE
    DESCRIPTION.

  Returns
  -------
  None.

  """
  
  weights_fp_stencil = []
  weights_fm_stencil = []
  batch_size = q_batch.shape[1]
  for i in range(batch_size):
    weights_fp_stencil.append(weno_i_weights(order, q_batch[0,i,:]))
    weights_fm_stencil.append(weno_i_weights(order, q_batch[1,i,:]))
    
  return numpy.array([weights_fp_stencil, weights_fm_stencil])

def weno_i_split(order, q):
  """
  Return WENO reconstruction at a given location

  Returns
  -------
  None.

  """
  
  weights = weno_i_weights(order, q)
  q_stencils = weno_i_stencils(order,q)
  return numpy.dot(weights, q_stencils)

# split the WENO method into computing stencils and then weights
def weno_stencils(order, q):
  """
  Compute WENO stencils
  
  Parameters
  ----------
  order : int
    The stencil width.
  q : numpy array
    Scalar data to reconstruct.

  Returns
  -------
  stencils

  """
  a = weno_coefficients.a_all[order]
  np = len(q) - 2 * order
  q_stencils = numpy.zeros((order, len(q)))
  for i in range(order, np+order):   
      for k in range(order):
          for l in range(order):
              q_stencils[k,i] += a[k, l] * q[i+k-l]
  
  return q_stencils

def weno_weights(order, q):
  """
  Compute WENO weights

  Parameters
  ----------
  order : int
    The stencil width.
  q : numpy array
    Scalar data to reconstruct.

  Returns
  -------
  stencil weights

  """
  C = weno_coefficients.C_all[order]
  sigma = weno_coefficients.sigma_all[order]

  beta = numpy.zeros((order, len(q)))
  w = numpy.zeros_like(beta)
  np = len(q) - 2 * order
  epsilon = 1e-16
  for i in range(order, np+order):
      alpha = numpy.zeros(order)
      for k in range(order):
          for l in range(order):
              for m in range(l+1):
                  beta[k, i] += sigma[k, l, m] * q[i+k-l] * q[i+k-m]
          alpha[k] = C[k] / (epsilon + beta[k, i]**2)
      w[:, i] = alpha / numpy.sum(alpha)
  
  return w

def weno_new(order, q):
  """
  Compute WENO reconstruction

  Parameters
  ----------
  order : int
    Stencil Width.
  q : TYPE
    Scalar data to reconstruct.

  Returns
  -------
  qL: numpy array
    Reconstructed data.

  """
  
  weights = weno_weights(order, q)
  q_stencils = weno_stencils(order, q)
  qL = numpy.zeros_like(q)
  np = len(q) - 2 * order
  for i in range(order, np+order):
    qL[i] = numpy.dot(weights[:, i], q_stencils[:,i])
    
  return qL

def RandomInitialCondition(grid: burgers.Grid1d, 
                           #seed: int = 44,
                           #offset: float = 0.0,
                           amplitude: float = 1.0,
                           k_min: int = 2,
                           k_max: int = 10):
    """ Generate random initial conditions """
    #rs = np.random.RandomState(seed)
    if k_min % 2 == 1:
      k_min += 1
    if k_max % 2 == 2:
      k_max += 1
    step = (k_max-k_min)/2
    k_values = numpy.arange(k_min, k_max,2)
    #print(k_values)
    k = numpy.random.choice(k_values, 1)
    b = numpy.random.uniform(-amplitude, amplitude, 1)
    a = 3.5 - numpy.abs(b)
    return  a + b * numpy.sin(k*numpy.pi*grid.x/(grid.xmax-grid.xmin))  
  
class WENOSimulation(burgers.Simulation):
    
    def __init__(self, grid, C=0.5, weno_order=3, timesteps=None):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.weno_order = weno_order
        
        ### GYM related
        self.current_state = None
        self.tmax_episode = 0.02*(self.grid.xmax - self.grid.xmin)/1.0
        self.timesteps_episode = timesteps

    def init_cond(self, type="tophat"):
        if type == "smooth_sine":
            self.grid.u = numpy.sin(2 * numpy.pi * self.grid.x)
        elif type == "gaussian":
            self.grid.u = 1.0 + numpy.exp(-60.0*(self.grid.x - 0.5)**2)
        elif type == "random":
            self.grid.u = RandomInitialCondition(self.grid)
        else:
            super().init_cond(type)
        self.grid.uactual[:] = self.grid.u[:] 

    def burgers_flux(self, q):
        return 0.5*q**2

    def rk_substep(self):
        
        # get the solution data
        g = self.grid
        
        # apply boundary conditions
        g.fill_BCs()
        
        #comput flux at each point
        f = self.burgers_flux(g.u)
        
        # get maximum velocity
        alpha = numpy.max(abs(g.u))
        
        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2
        
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        
        # compute fluxes at the cell edges
        # compute f plus to the right
        fpr[1:] = weno_new(self.weno_order, fp[:-1])
        # compute f minus to the left
        # pass the data in reverse order
        fml[-1::-1] = weno_new(self.weno_order, fm[-1::-1])
        
        # compute flux from fpr and fml
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()
        
        rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
        return rhs
    
    def rk_substep_actual(self):
        
        # get the solution data
        g = self.grid
        
        # apply boundary conditions
        g.fill_BCs()
        
        #comput flux at each point
        f = self.burgers_flux(g.uactual)
        
        # get maximum velocity
        alpha = numpy.max(abs(g.uactual))
        
        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.uactual) / 2
        fm = (f - alpha * g.uactual) / 2
        
        fpr = g.scratch_array()
        fml = g.scratch_array()
        flux = g.scratch_array()
        
        # compute fluxes at the cell edges
        # compute f plus to the right
        fpr[1:] = weno_new(self.weno_order, fp[:-1])
        # compute f minus to the left
        # pass the data in reverse order
        fml[-1::-1] = weno_new(self.weno_order, fm[-1::-1])
        
        # compute flux from fpr and fml
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()
        
        rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
        return rhs
    
    def rk_substep_loop(self):
        
        # get the solution data
        g = self.grid
        
        # apply boundary conditions
        g.fill_BCs()
        
        #comput flux at each point
        f = self.burgers_flux(g.u)
        
        # get maximum velocity
        alpha = numpy.max(abs(g.u))
        
        # Lax Friedrichs Flux Splitting
        fp = (f + alpha * g.u) / 2
        fm = (f - alpha * g.u) / 2
        
        fpr = g.scratch_array()
        fml = g.scratch_array()
        
        flux = g.scratch_array()
        
        w_o = self.weno_order
        np = len(f) - 2*w_o
        for i in range(w_o, np+w_o):
          #fp_stencil = fp[i-2:i+3]
          #fm_stencil = fm[i-3:i+2]
          fp_stencil = fp[i-w_o+1:i+w_o]
          fm_stencil = fm[i-w_o:i+w_o-1]
          
          fpr[i+1] = weno_i_split(self.weno_order, fp_stencil)
          fml[i] = weno_i_split(self.weno_order, fm_stencil)          
        
        # compute fluxes at the cell edges
        # compute f plus to the right
        #fpr[1:] = weno_new(self.weno_order, fp[:-1])
  
        
        # compute f minus to the left
        # pass the data in reverse order
        #fml[-1::-1] = weno_new(self.weno_order, fm[-1::-1])
        
        # compute flux from fpr and fml
        flux[1:-1] = fpr[1:-1] + fml[1:-1]
        rhs = g.scratch_array()
        
        rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
        return rhs


    def RK4(self, dt):
      """
      Compute one Runge-Kutta time step. Performs state transition for a discrete time step using standard WENO.
      
      Parameters
      ----------
      dt : float
        timestep.

      Returns
      -------
      Return state after taking one time step.

      """
      
      g = self.grid
      
      # fill the boundary conditions
      g.fill_BCs()
      
      # RK4
      # Store the data at the start of the step
      u_start = g.u.copy()
      k1 = dt * self.rk_substep()
      g.u = u_start + k1 / 2
      k2 = dt * self.rk_substep()
      g.u = u_start + k2 / 2
      k3 = dt * self.rk_substep()
      g.u = u_start + k3
      k4 = dt * self.rk_substep()
      g.u = u_start + (k1 + 2 * (k2 + k3) + k4) / 6

    def Euler(self, dt):
      """
      Compute one time step using explicit Euler time stepping. Performs state transition for a discrete time step using standard WENO.
      Euler time stepping is first order accurate in time. Should be run with a small time step.
      
      Parameters
      ----------
      dt : float
        timestep.

      Returns
      -------
      Return state after taking one time step.

      """
      
      g = self.grid
      
      # fill the boundary conditions
      g.fill_BCs()
      
      # RK4
      # Store the data at the start of the step
      u_start = g.u.copy()
      k1 = dt * self.rk_substep()
      g.u = u_start + k1

    def Euler_actual(self, dt):
      """
      Compute one time step using explicit Euler time stepping. Performs state transition for a discrete time step using standard WENO.
      Euler time stepping is first order accurate in time. Should be run with a small time step.
      
      Parameters
      ----------
      dt : float
        timestep.

      Returns
      -------
      Return state after taking one time step.

      """
      
      g = self.grid
      
      # fill the boundary conditions
      g.fill_BCs()
      
      # RK4
      # Store the data at the start of the step
      u_start = g.uactual.copy()
      k1 = dt * self.rk_substep_actual()
      g.uactual = u_start + k1
    
    def RK4_loop(self, dt):
      """
      Compute one time step. Takes the actions and performs state transition for a discrete time step.
      
      Parameters
      ----------
      dt : float
        timestep.

      Returns
      -------
      Return state after taking one time step.

      """
      
      g = self.grid
      
      # fill the boundary conditions
      g.fill_BCs()
      
      # RK4
      # Store the data at the start of the step
      u_start = g.u.copy()
      k1 = dt * self.rk_substep_loop()
      g.u = u_start + k1 / 2
      k2 = dt * self.rk_substep_loop()
      g.u = u_start + k2 / 2
      k3 = dt * self.rk_substep_loop()
      g.u = u_start + k3
      k4 = dt * self.rk_substep_loop()
      g.u = u_start + (k1 + 2 * (k2 + k3) + k4) / 6

    def evolve_timesteps(self):
        """ evolve the Burger equation using RK4"""
        self.t = 0.0
        
        # main evolution loop
        #while self.t < tmax:
        timesteps = self.timesteps_episode
        while timesteps > 0:
          
            # get the timestep
            # evolving with constant time step for testing. !!! CAUTION !!!
            dt = self.timestep()

            #if self.t + dt > tmax:
            #    dt = tmax - self.t

            #self.RK4(dt)
            self.Euler(dt)
            
            self.t += dt
            timesteps=timesteps-1
            
    def evolve_onestep(self):
        """ evolve the Burger equation using RK4"""
        self.t = 0.0
        
        # main evolution loop
        #while self.t < tmax:
          
        # get the timestep
        dt = self.timestep(self.C)

        #if self.t + dt > tmax:
        #    dt = tmax - self.t

        g = self.grid
      
        # fill the boundary conditions
        g.fill_BCs()
        
        # RK4
        # Store the data at the start of the step
        u_start = g.u.copy()
        k1 = dt * self.rk_substep()
        g.u = u_start + k1
        
        self.t += dt
        timesteps=timesteps-1
        
    def evolve_tmax(self, tmax):
        """ evolve the Burger equation using RK4"""
        self.t = 0.0
        timesteps=[]
        # main evolution loop
        while self.t < tmax:
          
            # get the timestep
            dt = self.timestep(self.C)

            if self.t + dt > tmax:
                dt = tmax - self.t

            #self.RK4_loop(dt)
            self.Euler(dt)
            
            self.t += dt
            timesteps.append(dt)
        print("computed for {} timesteps with average time step size {}".format(len(timesteps), numpy.mean(numpy.array(timesteps))))
    
    def prep_state(self):
      """
      Return state at current time step. Returns fpr and fml vector slices.

      Returns
      -------
      state: numpy array.
        State vector to be sent to the policy. 
        Size: 2 (one for fp and one for fm) BY grid_size BY stencil_size
        Example: order =3, grid = 128

      """
      # get the solution data
      g = self.grid
      
      # apply boundary conditions
      g.fill_BCs()
      
      #comput flux at each point
      f = self.burgers_flux(g.u)
      
      # get maximum velocity
      alpha = numpy.max(abs(g.u))
      
      # Lax Friedrichs Flux Splitting
      fp = (f + alpha * g.u) / 2
      fm = (f - alpha * g.u) / 2
      
      w_o = self.weno_order
      np = len(f) - 2*w_o
      fp_stencils = []
      fm_stencils = []
      
      for i in range(w_o, np+w_o):
        fp_stencil = fp[i-w_o+1:i+w_o]
        fp_stencils.append(fp_stencil)
        fm_stencil = fm[i-w_o:i+w_o-1]
        fm_stencils.append(fm_stencil)
        
      state = numpy.array([fp_stencils, fm_stencils])
      
      # save this state so that we can use it to compute next state
      self.current_state = state
      
      #print("size of state is: {}".format(numpy.shape(state)))
      return state
    
    def reset(self, inittype):
      """
      Reset the environment.Tasks here:
        Reset initial conditions.
        Set time to 0.

      Returns
      -------
      None.

      """
      
      self.t = 0.0
      if inittype == None:
        self.init_cond("random") 
      else:
        self.init_cond(inittype)
        
      return self.prep_state()
      
      
    def step(self, action):
      """
      Perform a single time step.

      Parameters
      ----------
      action : numpy array
        Weights for reconstructing WENO weights. This will be multiplied with the output from weno_stencils
        size: grid-points X order X 2
        Note: at each i+1/2 location we have an fpl and fmr.

      Returns
      -------
      state: numpy array.
        solution predicted using action
      reward: float
        scalar value - reward for current action
      done: boolean
        boolean signifying end of episode
      info : dictionary
        not passing anything now
      """
      
      done = False
      g = self.grid
      
      # Store the data at the start of the time step      
      u_start = g.u.copy()
      
      # get the timestep
      # passing self.C will cause this to take variable time steps
      # for now work with constant time step = 0.0005
      dt = self.timestep()

      # update current time
      #if self.t + dt > self.tmax_episode:
      #    dt = self.tmax_episode - self.t
      #    done = True
      #    print("number of time steps: {}".format(self.timesteps_episode+1))
      
      state = self.current_state
      
      q_stencils = weno_i_stencils_batch(self.weno_order, state)
      
      # storage
      fpr = g.scratch_array()
      fml = g.scratch_array()

      flux = g.scratch_array()
      
      w_o = self.weno_order
      np = len(flux) - 2*w_o
      for i in range(w_o, np+w_o):      
        # TODO vectorize with numpy.einsum ?
        fpr[i+1] = numpy.dot(action[0,i-w_o,:], q_stencils[0,i-w_o,:]) #weno_i_split(self.weno_order, fp_stencil)
        fml[i] = numpy.dot(action[1,i-w_o,:], q_stencils[1,i-w_o,:])#weno_i_split(self.weno_order, fm_stencil)          
      
      flux[1:-1] = fpr[1:-1] + fml[1:-1]
      rhs = g.scratch_array()
      
      rhs[1:-1] = 1/g.dx * (flux[1:-1] - flux[2:])
      
      k1 = dt * rhs
      g.u = u_start + k1
      
      #update the solution time
      self.t += dt
      self.timesteps_episode -= 1
      
      if self.timesteps_episode == 0:
        done = True
      
      #compute reward
      reward = 0.0
      self.Euler_actual(dt)
      
      error = numpy.max(numpy.abs(g.u[g.ilo:g.ihi+1]-g.uactual[g.ilo:g.ihi+1]))
      reward = -numpy.log(error)
      
      # should this reward be clipped?
      if reward < 10:
        reward = 0
      
      return self.prep_state(), reward, done, None

def test_evolve(args):
  xmin = args["xmin"]
  xmax = args["xmax"]
  nx = args["nx"]
  order = args["order"]
  ng = args["ng"]
  
  timesteps = args["timesteps"]
  
  g = burgers.Grid1d(nx, ng, bc="periodic")
  # maximum evolution time based on period for unit velocity
  tmax = (xmax - xmin)/1.0
  
  C = args["C"]
  
  pyplot.clf()
  
  s = WENOSimulation(g, C, order, timesteps)
  s.init_cond(args["inittype"])

  uinit = s.grid.u.copy()

  #tic = time.perf_counter()
  #s.evolve_tmax(tmax)
  #toc = time.perf_counter()
  #print(f"Time for one episode is {toc - tic:0.4f} seconds")

  uinit = s.grid.u.copy()

  s.evolve_timesteps()
  #fig,ax = pyplot.add_subplot()

  #pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color="k", label="Final State")
  
  #pyplot.plot(g.x[g.ilo:g.ihi+1], uinit[g.ilo:g.ihi+1], ls=":", color="k", zorder=-1, label="Initial State")
  
  #pyplot.xlabel("$x$")
  #pyplot.ylabel("$u$")
  #pyplot.show()
  return s.grid.u
    
      
def test_environment(args):
  xmin = args["xmin"]
  xmax = args["xmax"]
  nx = args["nx"]
  order = args["order"]
  ng = args["ng"]
  
  timesteps = args["timesteps"]
  g = burgers.Grid1d(nx, ng, bc="periodic")
  
  # maximum evolution time based on period for unit velocity
  tmax = (xmax - xmin)/1.0
  
  C = args["C"]
  
  pyplot.clf()
  
  s = WENOSimulation(g, C, order, timesteps)

  uinit = s.grid.u.copy()
  
  done = False
  state = s.reset(args["inittype"])
  rewards = []
  while done == False:
    actions = weno_i_weights_batch(order, state) # state.shape = (2 (fp, fm), # grid, stencil_size )
    state, reward, done, info = s.step(actions) #action.shape = (2 (fp, fm), # grid, weight_vector_length )
    rewards.append(reward)
  
  #pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color="r")
  #pyplot.plot(g.x[g.ilo:g.ihi+1], g.uactual[g.ilo:g.ihi+1], ls=":", color="k")
  #pyplot.plot(g.x[g.ilo:g.ihi+1], uinit[g.ilo:g.ihi+1], ls=":", color="r", zorder=-1, label="Initial State")
    
  #pyplot.xlabel("$x$")
  #pyplot.ylabel("$u$")
  #pyplot.show()
  return s.grid.u, rewards


if __name__ == "__main__":
    xmin = 0.0
    xmax = 1.0
    nx = 128
    order = 3
    ng = order+1
    C = 0.1
    inittype = "sine"
    timesteps = 300
  
    args ={}
    args["xmax"] = xmin
    args["xmin"] = xmin
    args["nx"] = nx
    args["order"] = order
    args["ng"] = ng
    args["C"] = C
    args["inittype"] = inittype
    args["timesteps"] = timesteps
  
    g = burgers.Grid1d(nx, ng, bc="periodic")
    #act = test_evolve(args)
    pred, rewards = test_environment(args)
    #pyplot.plot(act[g.ilo:g.ihi+1], label = "actual")
    #pyplot.plot(pred[g.ilo:g.ihi+1],ls=":", color="k", label = "env")
    #pyplot.legend()
    pyplot.plot(rewards)
    #pyplot.plot(numpy.abs(act[g.ilo:g.ihi+1]-pred[g.ilo:g.ihi+1]))
    #assert(numpy.allclose(act[g.ilo:g.ihi+1], pred[g.ilo:g.ihi+1], atol = 0.0, rtol = 1e-02))

    '''
    #-----------------------------------------------------------------------------
    # sine
    
    xmin = 0.0
    xmax = 1.0
    nx = 128
    order = 3
    ng = order+1
    g = burgers.Grid1d(nx, ng, bc="periodic")
    
    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin)/1.0
    
    C = 0.05
    
    pyplot.clf()
    
    s = WENOSimulation(g, C, order)
    s.init_cond("sine")

    uinit = s.grid.u.copy()

    #tic = time.perf_counter()
    #s.evolve_tmax(tmax)
    #toc = time.perf_counter()
    #print(f"Time for one episode is {toc - tic:0.4f} seconds")

    g = s.grid
    #pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color='k', label="Final State")
    
    for i in range(0, 10):
        tend = (i+1)*0.02*tmax
        s.init_cond("sine")
    
        uinit = s.grid.u.copy()
    
        s.evolve_tmax(tend)
    
        c = 1.0 - (0.1 + i*0.1)
        g = s.grid
        pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color=str(c))
    
    
    g = s.grid
    pyplot.plot(g.x[g.ilo:g.ihi+1], uinit[g.ilo:g.ihi+1], ls=":", color="k", zorder=-1, label="Initial State")
    
    pyplot.xlabel("$x$")
    pyplot.ylabel("$u$")
    pyplot.show()
    #pyplot.savefig("weno-burger-sine.pdf")
    
    
    #state = s.reset()
    
    
    
    # Compare the WENO and "standard" (from burgers.py) results at low res
    nx = 64
    tend = 0.2
    g_hires = burgers.Grid1d(512, ng, bc="periodic")
    s_hires = WENOSimulation(g_hires, C, order)
    s_hires.init_cond("sine")
    s_hires.evolve(tend)
    gW3 = burgers.Grid1d(nx, 4, bc="periodic")
    sW3 = WENOSimulation(gW3, C, 3)
    sW3.init_cond("sine")
    sW3.evolve(tend)
    gW5 = burgers.Grid1d(nx, 6, bc="periodic")
    sW5 = WENOSimulation(gW5, C, 5)
    sW5.init_cond("sine")
    sW5.evolve(tend)
    g = burgers.Grid1d(nx, ng, bc="periodic")
    s = burgers.Simulation(g)
    s.init_cond("sine")
    s.evolve(C, tend)
    pyplot.clf()
    pyplot.plot(g_hires.x[g_hires.ilo:g_hires.ihi+1], 
                g_hires.u[g_hires.ilo:g_hires.ihi+1], 'k--', label='High resolution')
    pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], 'gd', label='PLM, MC')
    pyplot.plot(gW3.x[gW3.ilo:gW3.ihi+1], gW3.u[gW3.ilo:gW3.ihi+1], 'bo', label='WENO, r=3')
    pyplot.plot(gW5.x[gW5.ilo:gW5.ihi+1], gW5.u[gW5.ilo:gW5.ihi+1], 'r^', label='WENO, r=5')
    pyplot.xlabel("$x$")
    pyplot.ylabel("$u$")
    pyplot.legend()
    pyplot.xlim(0.5, 0.9)
    pyplot.legend(frameon=False)
    pyplot.savefig("weno-vs-plm-burger.pdf")
    
    
    
    #-----------------------------------------------------------------------------
    # rarefaction
    
    xmin = 0.0
    xmax = 1.0
    nx = 256
    order = 3
    ng = order+1
    g = burgers.Grid1d(nx, ng, bc="outflow")
    
    # maximum evolution time based on period for unit velocity
    tmax = (xmax - xmin)/1.0
    
    C = 0.5
    
    pyplot.clf()
    
    s = WENOSimulation(g, C, order)
    
    for i in range(0, 10):
        tend = (i+1)*0.02*tmax
    
        s.init_cond("rarefaction")
    
        uinit = s.grid.u.copy()
    
        s.evolve(tend)
    
        c = 1.0 - (0.1 + i*0.1)
        pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color=str(c))
    
    
    pyplot.plot(g.x[g.ilo:g.ihi+1], uinit[g.ilo:g.ihi+1], ls=":", color="0.9", zorder=-1)
    
    pyplot.xlabel("$x$")
    pyplot.ylabel("$u$")
    
    pyplot.savefig("weno-burger-rarefaction.pdf")

    #-----------------------------------------------------------------------
    # Convergence test at t = 0.1 using gaussian data
    
    
    problem = "gaussian"

    xmin = 0.0
    xmax = 1.0
    tmax = 0.05
    orders = [3, 4]
    N = [64, 81, 108, 128, 144, 192, 256]
    #N = 2**numpy.arange(5,10)
    C = 0.5

    errs = []

    colors="brcg"

    for order in orders:
        ng = order+1
        errs.append([])
        for nx in N:
            print(order, nx)
            gu = burgers.Grid1d(nx, ng, xmin=xmin, xmax=xmax)
            su = WENOSimulation(gu, C=0.5, weno_order=order)
        
            su.init_cond("gaussian")
        
            su.evolve(tmax)
            
            uexact = burgers_sine_exact(gu.x, tmax)
        
            errs[-1].append(gu.norm(gu.u - uexact))
    
    pyplot.clf()
    N = numpy.array(N, dtype=numpy.float64)
    for n_order, order in enumerate(orders):
        pyplot.scatter(N, errs[n_order],
                       color=colors[n_order],
                       label=r"WENO, $r={}$".format(order))
    pyplot.plot(N, errs[0][-2]*(N[-2]/N)**(5),
                linestyle="--", color=colors[0],
                label=r"$\mathcal{{O}}(\Delta x^{{{}}})$".format(5))
    pyplot.plot(N, errs[1][-3]*(N[-3]/N)**(7),
                linestyle="--", color=colors[1],
                label=r"$\mathcal{{O}}(\Delta x^{{{}}})$".format(7))

    ax = pyplot.gca()
    ax.set_ylim(numpy.min(errs)/5, numpy.max(errs)*5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    pyplot.xlabel("N")
    pyplot.ylabel(r"$\| a^\mathrm{final} - a^\mathrm{init} \|_2$",
               fontsize=16)
    pyplot.title("Convergence of Burger's, Gaussian, RK4")

    pyplot.legend(frameon=False)
    pyplot.savefig("weno-converge-burgers.pdf")
    '''
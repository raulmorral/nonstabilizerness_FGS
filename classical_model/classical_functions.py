import numpy as np
import sys
import os
import pickle


def create_path(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed. Might already exist!" % path)
    else:
        print ("Successfully created the directory %s " % path)

# Functions to save and load python type objects to file using pickle.
def save_obj(data, filename ):
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename ):
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)
    
class site(object):
    '''
    A site is characterized by a position in a 1-d chain. Each site contains exactly two particles.
    '''
    def __init__(self, position):

        self.x = position

        self.particle1 = None
        self.particle2 = None

    def set_particles(self, particle1, particle2):
        self.particle1 = particle1
        self.particle2 = particle2

        particle1.site = self
        particle2.site = self
    
    def which_particles(self):
        return (self.particle1, self.particle2)
    
    def __str__(self):
        return f'Site at position {self.x} containing particles {self.particle1} and {self.particle2}' 
    def __repr__(self):
        return str(self.x)


class color(object):
    '''
    A site is characterized by a position in a 1-d chain. Each site contains exactly two particles.
    '''
    def __init__(self, c):

        self.c = c

        self.particles = []
    
    def add_particles(self, particles):
        for p in particles:
            self.particles.append(p)
            p.color = self

    def empty(self):
        self.particles = []
    
    def __str__(self):
        if len(self.particles)==0:
            return f'No particles with color {self.c}'
        st = f'Color {self.c} containing particles '
        for p in particles:
            st += f'{self.particle1}, '

        return st

        
    def __repr__(self):
        return str(self.c)

    def __eq__(self, other):
        return self.c == other.c

class particle(object):
    '''
    Particle with some properties.
    '''
    def __init__(self, name):
        
        self.particle_id = name

        self.site = None
        self.color = None

    def set_color(self, c):
        self.color = c

        c.particles.append(self)
    
    def __str__(self):
        return f'{self.particle_id}'


class lattice(object):
    '''
    A lattice with L sites, each of them containing 2 particles.
    '''
    def __init__(self, L):

        self.sites = [site(ii) for ii in range(L)]
        self.colors = [color(ii) for ii in range(L)]

        for ii, s in enumerate(self.sites):
            p1 = particle(2*ii)
            p2 = particle(2*ii+1)
            s.set_particles(p1, p2)
            self.colors[ii].add_particles([p1,p2])

        self.L = L


        self.empty_colors = []

    def __str__(self):
        x = ''
        for s in self.sites:
            x += f'Site at position {s.x} containing particles {s.particle1} (color {s.particle1.color.c}) and {s.particle2} (color {s.particle2.color.c})\n' 
        return (x)
        
    def braiding_gate(self, i, j):
        '''
        Exchanges randomly the positions of the particles in sites i and j.
        '''
        order = np.arange(4)
        np.random.shuffle(order)
        particles_ij = [self.sites[i].particle1, self.sites[i].particle2, self.sites[j].particle1, self.sites[j].particle2]
        self.sites[i].set_particles(particles_ij[order[0]], particles_ij[order[1]])
        self.sites[j].set_particles(particles_ij[order[2]], particles_ij[order[3]])

    def braiding_general(self):
        '''
        Applies a random braiding gate in the whole system.
        '''
        order = np.arange(2*self.L)
        np.random.shuffle(order)
        ptcls = []
        for ii in range(self.L):
            ptcls.append(self.sites[ii].particle1)
            ptcls.append(self.sites[ii].particle2)
        for ii in range(self.L):
            self.sites[ii].set_particles(ptcls[order[2*ii]], ptcls[order[2*ii+1]])
    
    
    def T_gate(self, i):
        '''
        Sets the two particles in site i to have the same color.
        '''
        if self.sites[i].particle1.color == self.sites[i].particle2.color:
            pass
        else:
            c1 = self.sites[i].particle1.color
            c2 = self.sites[i].particle2.color
            
            for p in c2.particles:
                p.set_color(c1)

            c2.empty()
            self.empty_colors.append(c2.c)

    def measurement(self, i):
        if self.sites[i].particle1.color == self.sites[i].particle2.color:
            if len(self.sites[i].particle1.color.particles) == 2:
                pass
            else:
                c1 = self.sites[i].particle1.color
                c2 = self.colors[self.empty_colors[-1]]
                self.empty_colors.pop()

                particles = c1.particles

                c1.empty()

                for p in particles:
                    if p.site.x == i:
                        p.set_color(c1)
                    else:
                        p.set_color(c2)
                
        else:
            c1 = self.sites[i].particle1.color
            c2 = self.sites[i].particle2.color
            particles = c1.particles + c2.particles

            c1.empty()
            c2.empty()
            
            for p in particles:
                if p.site.x == i:
                    p.set_color(c1)
                else:
                    p.set_color(c2)

            

            
    def nullity(self):
        return len(self.empty_colors)


def undo_single_T(L, p):
        

    state = lattice(L)
    state.braiding_general()

    
    state.T_gate(0)
    
    
    T = 0
    
    while state.nullity() > 0:
        
        for jj in range(L//2):
            state.braiding_gate(2*jj, 2*jj+1)
            
        for jj in range(L):
            if np.random.rand() < p:
                state.measurement(jj)
                
        T += 1
        
        if state.nullity() > 0:
            for jj in range(L//2-1):
                state.braiding_gate(2*jj+1, 2*jj+2)

            for jj in range(L):
                if np.random.rand() < p:
                    state.measurement(jj)
                    

            T += 1
            
    return T
    
    
def undo_single_T_manytimes(L, p, case, reps):

    filename = f'/space/ge24yov/gaussian_nonstabilizerness_disentangle_classical//results_singleT_p'+str("%.2f"%p)+f'_L{L}_init_state_random_case{case}'
    
    times = []
    
    for ii in range(reps):
        times.append(undo_single_T(L, p))
        save_obj(times, filename)



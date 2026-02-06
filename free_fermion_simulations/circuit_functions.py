import numpy as np
from free_fermion_methods_cov import *


def circuit_model(L, p, T_therm, T_steady, case, initial_state = 'product'):
        
    if initial_state == 'product':
        state = product_0_FGS(L)
        filename = f'/space/ge24yov/gaussian_nonstabilizerness//results_mipt_p'+str("%.2f"%p)+f'_L{L}_case{case}'
    elif initial_state == 'random':
        state = random_FGS(L)
        filename = f'/space/ge24yov/gaussian_nonstabilizerness//results_mipt_p'+str("%.2f"%p)+f'_L{L}_init_state_random_case{case}'
    
    ent_entropies_ev = np.zeros(T_therm+1)
    nullity_ev = np.zeros(T_therm+1)
    
    count = 0
    ents = state.compute_entanglement_entropy(L//2-1)
    ent_entropies_ev[count] = ents
    nullity_ev[count] = stabilizer_nullity(state.Cov)
    
    for ii in range(T_therm):
        
        for jj in range(L//2):
            state.apply_U_bond(random_braiding(), 2*jj)
            
        state.apply_T_gate(L//2-1)
        state.apply_T_gate(L//2)
            
        for jj in range(L):
            if np.random.rand() < p:
                state.apply_measurement_particle_number(jj)
        
        for jj in range(L//2-1):
            state.apply_U_bond(random_braiding(), 2*jj+1) 
            
        state.apply_T_gate(L//2-1)
        state.apply_T_gate(L//2)
            
        for jj in range(L):
            if np.random.rand() < p:
                state.apply_measurement_particle_number(jj)
                
        count += 1
        ent_entropies_ev[count] = state.compute_entanglement_entropy(L//2-1)
        nullity_ev[count] = stabilizer_nullity(state.Cov)
            
            
    profile_av = np.zeros(L+1)
    nullity_av = 0
    
    for ii in range(T_steady):
        
        for jj in range(L//2):
            state.apply_U_bond(random_braiding(), 2*jj)
            
        state.apply_T_gate(L//2-1)
        state.apply_T_gate(L//2)
            
        for jj in range(L):
            if np.random.rand() < p:
                state.apply_measurement_particle_number(jj)
            
        profile_av += state.compute_profile(1)
        nullity_av += stabilizer_nullity(state.Cov)
        
        
        for jj in range(L//2-1):
            state.apply_U_bond(random_braiding(), 2*jj+1) 
            
        state.apply_T_gate(L//2-1)
        state.apply_T_gate(L//2)   
            
        for jj in range(L):
            if np.random.rand() < p:
                state.apply_measurement_particle_number(jj)
                
        profile_av += state.compute_profile(1)
        nullity_av += stabilizer_nullity(state.Cov)

#     return nullity_ev, ent_entropies_ev, ent_av / (T_steady+1), nullity_av / (T_steady+1)

    data = {}
    
    data['L'] = L
    data['num_T_gates'] = 2
    data['T_therm'] = T_therm+1
    data['initial_state'] = initial_state
    data['run_number'] = case
    
    data['T_steady'] = 2*T_steady
    
    data['ent_entropies_evolution'] = ent_entropies_ev
    data['nullity_evolution'] = nullity_ev
    data['profile_steady'] = profile_av / (2*T_steady)
    data['nullity_steady'] = nullity_av / (2*T_steady)
    
    
    data['covariance_matrix_last_step'] = state.Cov
         
    save_obj(data, filename)

def undo_random_FGS(L, p, case):
        

    state = random_FGS(L)
    filename = f'/space/ge24yov/gaussian_nonstabilizerness_disentangle//results_p'+str("%.2f"%p)+f'_L{L}_init_state_random_case{case}'
    
    
    #ents = state.compute_entanglement_entropy(L//2-1)
    #ent_entropies_ev = [ents]
    nullity_ev = [stabilizer_nullity(state.Cov)]
    
    while nullity_ev[-1] > 0:
        
        for jj in range(L//2):
            state.apply_U_bond(random_braiding(), 2*jj)
            
        for jj in range(L):
            if np.random.rand() < p:
                state.apply_measurement_particle_number(jj)
        
        for jj in range(L//2-1):
            state.apply_U_bond(random_braiding(), 2*jj+1) 

        for jj in range(L):
            if np.random.rand() < p:
                state.apply_measurement_particle_number(jj)
                
       #ent_entropies_ev.append(state.compute_entanglement_entropy(L//2-1))
        nullity_ev.append(stabilizer_nullity(state.Cov))
            
            
  

    data = {}
    
    data['L'] = L
    data['run_number'] = case
        
    #data['ent_entropies_evolution'] = ent_entropies_ev
    data['nullity_evolution'] = nullity_ev
             
    save_obj(data, filename)


def undo_single_T(L, p, case):
        

    state = random_braiding_state(L)
    
    state.apply_T_gate(0)
    state.update_covariance_matrix()
    
    
    T = 0
    
    while stabilizer_nullity(state.Cov) > 0:
        
        for jj in range(L//2):
            state.apply_U_bond(random_braiding(), 2*jj)
            
        for jj in range(L):
            if np.random.rand() < p:
                state.apply_measurement_particle_number(jj)
                
        T += 1
        
        if stabilizer_nullity(state.Cov) > 0:
            for jj in range(L//2-1):
                state.apply_U_bond(random_braiding(), 2*jj+1) 

            for jj in range(L):
                if np.random.rand() < p:
                    state.apply_measurement_particle_number(jj)
                    

            T += 1
            
    return T
    
    
def undo_single_T_manytimes(L, p, case, reps):

    filename = f'/space/ge24yov/gaussian_nonstabilizerness_disentangle//results_singleT_p'+str("%.2f"%p)+f'_L{L}_init_state_random_case{case}'
    
    times = []
    
    for ii in range(reps):
        times.append(undo_single_T(L, p, case))
        save_obj(times, filename)
        
    

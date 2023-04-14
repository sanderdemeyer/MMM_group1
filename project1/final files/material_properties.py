class EM_properties:
    def __init__(self, material):
        self.eps_0 = 8.85*10**(-12)
        self.mu_0 = 1.25663706*10**(-6)
        self.material = material
        if material == 'Copper':
            self.epsilon_r = 10**9 # should be infinite
            self.mu_r = 0.999994
            self.sigma = 58*10**6
        elif material == 'Silicon':
            self.epsilon_r = 11.68
            self.mu_r = 0.99837
            self.sigma = 1/(2.3*10**3)
        elif material == 'Silica':
            self.epsilon_r = 3
            self.mu_r = 0.99837
            self.sigma = 0
        elif material == 'vacuum':
            self.epsilon_r = 1
            self.mu_r = 1
            self.sigma = 0
        else:
            print('invalid material')
    def set_properties(self, epsilon, mu, sigma, left, right):
        epsilon[left:right,:] = self.eps_0*self.epsilon_r
        mu[left:right,:] = self.mu_0*self.mu_r
        sigma[left:right,:] = self.sigma
        return [epsilon, mu, sigma]
    
class material_grid:
    def __init__(self, material_list):
        # each element of material_list is [material, left, right, color]
        # material should be one of the defined materials in the class EM_properties
        # left and right are respectively the left and right edge of the block in the x-direction
        # color is optional. If not given, this is set to red. Diversify for the animation.
        self.eps_0 = 8.85*10**(-12)
        self.mu_0 = 1.25663706*10**(-6)
        for i, mat in enumerate(material_list):
            if len(mat) == 3:
                material_list[i].append('red')
        self.material_list = material_list
    def set_properties(self, epsilon, mu, sigma):
        for el in self.material_list:
            material = EM_properties(el[0])
            epsilon[el[1]:el[2],:] = self.eps_0*material.epsilon_r
            mu[el[1]:el[2],:] = self.mu_0*material.mu_r
            sigma[el[1]:el[2],:] = material.sigma
        return [epsilon, mu, sigma]

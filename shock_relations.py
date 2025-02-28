import math as m

pi = m.pi

gamma = eval(input("Enter gamma of air:"))
mach_1 = eval(input("Enter the upstream mach number:"))
shock_angle_deg = eval(input("Enter the shock angle in degrees:"))

shock_angle_rad = shock_angle_deg*(pi/180)
mach_1_n = mach_1*(m.sin(shock_angle_rad))

def calculate_def_angle(mach,shock_angle,gamma_):
    a = (mach*m.sin(shock_angle))**2 - 1
    b = 2+(mach**2)*(gamma_+m.cos(2*shock_angle))
    c = 2*(1/m.tan(shock_angle))
    d = c*(a/b)
    theta = m.atan(d)
    return theta

def_angle_1 = calculate_def_angle(mach_1,shock_angle_rad,gamma)

def get_mach_n_2(mach_n,gamma_):
    a = 1+((gamma_-1)/2)*(mach_n**2)
    b = gamma_*(mach_n**2)-((gamma_-1)/2)
    c = a/b
    mach_n_2 = m.sqrt(c)
    return mach_n_2

mach_2_n = get_mach_n_2(mach_1_n,gamma)
mach_2 = mach_2_n/(m.sin(shock_angle_rad-def_angle_1))

def get_static_density_ratio(mach_n,gamma_):
    a = (gamma_+1)*(mach_n**2)
    b = 2+(gamma_-1)*(mach_n**2)
    c = a/b
    return c

def get_static_pressure_ratio(mach_n,gamma_):
    a = 2*gamma_*(mach_n**2-1)
    b = gamma_+1
    c = 1+(a/b)
    return c

static_pressure_ratio = get_static_pressure_ratio(mach_1_n,gamma)
static_density_ratio = get_static_density_ratio(mach_1_n,gamma)

static_temperature_ratio = static_pressure_ratio/static_density_ratio

print("Mach number downstream:", mach_2)
print("Deflection angle:",def_angle_1)
print("Static pressure ratio P2/P1 across shock:",static_pressure_ratio)
print("Static density ratio P2/P1 across shock:",static_density_ratio)
print("Static temperature ratio P2/P1 across shock:",static_temperature_ratio)









    


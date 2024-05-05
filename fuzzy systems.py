import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Antecedents (inputs)
distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
car_speed = ctrl.Antecedent(np.arange(0, 101, 1), 'car_speed')

# Consequent (output)
acceleration = ctrl.Consequent(np.arange(0, 101, 1), 'acceleration')

# Define membership functions
distance['very_close'] = fuzz.trimf(distance.universe, [0, 0, 20])
distance['close'] = fuzz.trimf(distance.universe, [10, 30, 50])
distance['medium'] = fuzz.trimf(distance.universe, [40, 60, 80])
distance['far'] = fuzz.trimf(distance.universe, [70, 90, 100])

car_speed['slow'] = fuzz.trimf(car_speed.universe, [0, 0, 40])
car_speed['medium'] = fuzz.trimf(car_speed.universe, [30, 50, 70])
car_speed['fast'] = fuzz.trimf(car_speed.universe, [60, 100, 100])

acceleration['decelerate'] = fuzz.trimf(acceleration.universe, [0, 0, 50])
acceleration['maintain'] = fuzz.trimf(acceleration.universe, [40, 50, 60])
acceleration['accelerate'] = fuzz.trimf(acceleration.universe, [50, 100, 100])

# Define rules
rule1 = ctrl.Rule(distance['very_close'] & car_speed['slow'], acceleration['decelerate'])
rule2 = ctrl.Rule(distance['close'] & car_speed['slow'], acceleration['maintain'])
rule3 = ctrl.Rule(distance['medium'] & car_speed['slow'], acceleration['accelerate'])
rule4 = ctrl.Rule(distance['far'] & car_speed['slow'], acceleration['accelerate'])

rule5 = ctrl.Rule(distance['very_close'] & car_speed['medium'], acceleration['decelerate'])
rule6 = ctrl.Rule(distance['close'] & car_speed['medium'], acceleration['decelerate'])
rule7 = ctrl.Rule(distance['medium'] & car_speed['medium'], acceleration['maintain'])
rule8 = ctrl.Rule(distance['far'] & car_speed['medium'], acceleration['accelerate'])

rule9 = ctrl.Rule(distance['very_close'] & car_speed['fast'], acceleration['decelerate'])
rule10 = ctrl.Rule(distance['close'] & car_speed['fast'], acceleration['decelerate'])
rule11 = ctrl.Rule(distance['medium'] & car_speed['fast'], acceleration['decelerate'])
rule12 = ctrl.Rule(distance['far'] & car_speed['fast'], acceleration['maintain'])

# Create control system
acceleration_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])
car = ctrl.ControlSystemSimulation(acceleration_ctrl)

# Input values
car.input['distance'] = 30
car.input['car_speed'] = 60

# Compute the result
car.compute()

# Output value
print("Acceleration:", car.output['acceleration'])

# Plotting the membership functions
distance.view()
car_speed.view()
acceleration.view()

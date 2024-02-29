from sympy import symbols, integrate, Abs, solve

# Define symbols
x, alpha = symbols('x alpha')

# Define the integral expressions for each interval
integral_1_expression = Abs(4*x**2/alpha**2 - 0.5)
integral_2_expression = Abs(4*(x-alpha)**2/alpha**2 - 0.5)

# Perform the integrations over the specified intervals
integral_1_result = integrate(integral_1_expression, (x, 0, alpha/2))
integral_2_result = integrate(integral_2_expression, (x, alpha/2, alpha))

# Calculate the index of fuzziness nu(A)
nu_A_result = (integral_1_result + integral_2_result) / alpha
nu_A_result.simplify()
print(nu_A_result)
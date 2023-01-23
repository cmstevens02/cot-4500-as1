import numpy as np

# takes in a binary string 
# returns the double precision decimal form
def binaryToFloat(num):
    # get the sign, expo, and mantissa from the string
    sign = int(num[0], 2)
    expo = int(num[1:12], 2)  #issue
    mantissaStr = num[12:]

    # process the mantissa 
    mantissa = 0; 
    for x in range(1,len(mantissaStr)+1): 
        mantissa += (0.5)**x * int(mantissaStr[x-1])


    # put it together to get the number! yayy
    res = ((-1)**sign) * ((2)**(expo-1023)) * (1+mantissa)

    return res

# to use for normalizing numbers
def getNumOfDigits(n):
    whole = n // 1
    return len(str(int(whole)))

# num is the number to be chopped, k is the digit to chop to
def chop(num, k):
    expo = getNumOfDigits(num)
    normalized = num / 10**expo
    toChop = normalized * 10**k 
    removeDecimals = int(toChop // 1)
    backToNormalized = removeDecimals / 10**k
    chopped = backToNormalized * (10**expo)
    return round(chopped, k-expo)

# num is the number to be rounded, k is the digit to round to
def myRound(num, k):
    #print("num:", num)
    expo = getNumOfDigits(num)
    normalized = num / 10**expo
    toRound = normalized * 10**(k+1) + 5
    backToNormalized = (toRound // 1) / 10**(k+2)
    rounded = backToNormalized * 10**(expo + 1)
    return round(rounded, k-expo)

def absolute_error(precise, approximate): 
    op = precise-approximate
    return abs(op)

def relative_error(precise, approximate):
    op = abs(precise - approximate) / abs(precise)
    return op


# referenced from class 
def find_min_terms(func): 

    if (func == "(-1**k) * (x**k) / (k**3)"):
        tol = 0.0001
        flipped_tol = 1 / tol
        n = flipped_tol / 10**3
        return(int(n))
 
def bisection_method(left: float, right: float, given_function: str):
    # pre requisites
    # 1. we must have the two ranges be on opposite ends of the function (such that
    # function(left) and function(right) changes signs )
    x = left
    intial_left = eval(given_function)
    x = right
    intial_right = eval(given_function)
    if intial_left * intial_right >= 0:
        print("Invalid inputs. Not on opposite sides of the function")
        return

    tolerance: float = .0001
    diff: float = right - left

    # we can only specify a max iteration counter (this is ideal when we dont have all
    # the time in the world to find an exact solution. after 10 iterations, lets say, we
    # can approximate the root to be ###)
    max_iterations = 20
    iteration_counter = 0
    while (diff >= tolerance):
        iteration_counter += 1

        # find function(midpoint)
        mid_point = (left + right) / 2
        x = mid_point
        evaluated_midpoint = eval(given_function)

        if evaluated_midpoint == 0.0:
            break
        
        # find function(left)
        x = left
        evaluated_left_point = eval(given_function)
        
        # this section basically checks if we have crossed the origin point (another way
        # to describe this is if f(midpoint) * f(left_point) changed signs)
        first_conditional: bool = evaluated_left_point < 0 and evaluated_midpoint > 0
        second_conditional: bool = evaluated_left_point > 0 and evaluated_midpoint < 0

        if first_conditional or second_conditional:
            right = mid_point
        else:
            left = mid_point
        
        diff = abs(right - left)

        # OPTIONAL: you can see how the root finding for bisection works per iteration
    return iteration_counter

# also referenced from class, thanks Prof Parra 
def custom_derivative(value):
    return (3 * value* value) + (8 * value)


def newton_raphson(initial_approximation: float, tolerance: float, sequence: str):
    # remember this is an iteration based approach...
    iteration_counter = 0

    # finds f
    x = initial_approximation
    f = eval(sequence)

    # finds f' 
    f_prime = custom_derivative(initial_approximation)
    
    approximation: float = f / f_prime
    while(abs(approximation) >= tolerance):
        # finds f
        x = initial_approximation
        f = eval(sequence)

        # finds f' 
        f_prime = custom_derivative(initial_approximation)

        # division operation
        approximation = f / f_prime

        # subtraction property
        initial_approximation -= approximation
        iteration_counter += 1

    return iteration_counter

if __name__ == "__main__":

# 1) Using double precision, calculate the resulting values (format to 5 decimal places)
    binaryStr = "010000000111111010111001"

    double = binaryToFloat(binaryStr)
    print(double, end="\n\n")

# 2) Repeat exercise 1 using three-digit chopping arithmetic
    chopIt = chop(double, 3)
    print(chopIt, end="\n\n")


# 3) Repeat exercise 1 using three-digit rounding arithmetic 
    roundIt = myRound(double, 3)
    print(roundIt, end="\n\n")

# 4) Compute the absolute and relative error with the exact value from question 1 and its 3 digit rounding 
    abs_error = absolute_error(double,roundIt)
    print(abs_error)

    rel_error = relative_error(double, roundIt)
    print(rel_error, end="\n\n")

# 5) What is the minimum number of terms needed to compute f(1) with error < 10^-4?
alt = "(-1**k) * (x**k) / (k**3)"
print(find_min_terms(alt), end="\n\n")

# 6) num of iterations to solve f(x) = x^3 + 4x^2 - 10 = 0 with 10^-4 accuracy, a = -4 b = 7

# bisection method
left = -4
right = 7
function_string = "x**3 + (4*(x**2)) - 10"
print(bisection_method(left, right, function_string), end="\n\n")

#newton Raphson method
initial_approximation: float = -4
tolerance: float = .0001
print(newton_raphson(initial_approximation, tolerance, function_string))
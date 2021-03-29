from homework1_twgoon import *
import numpy as np

a1 = np.array([
    [1, 2, 3],
    [3, 4, 5]
])

a2 = np.array([
    [2, 2, 2],
    [2, 2, 2]
])

a3 = np.array([
    [24, 2, 0],
    [1, 20, 8]
])

a4 = np.array([
    [6, 3],
    [7, 1]
])

a5 = np.array([
    [9, 3],
    [7, 1],
    [1, 2]
])

a6 = np.array([
    [6, 3, 5],
    [7, 1, 8],
    [1, 1, 1]
])

def main():
    print("Test Problem 1: ", test_prob_1())
    print("Test Problem 2: ", test_prob_2())
    print("Test Problem 3: ", test_prob_3())
    print("Test Problem 4: ", test_prob_4())
    print("Test Problem 5: ", test_prob_5())
    print("Test Problem 6: ", test_prob_6())
    print("Test Problem 7: ", test_prob_7())
    print("Test Problem 8: ", test_prob_8())
    print("Test Problem 9: ", test_prob_9())
    print("Test Problem 10: ", test_prob_10())
    print("Test Problem 11: ", test_prob_11())
    print("Test Problem 12: ", test_prob_12())
    print("Test Problem 13: ", test_prob_13())

def test_prob_1():
    """Test problem 1"""

    r = problem1(a1, a2)

    sol = np.array([
        [3, 4, 5],
        [5, 6, 7]
    ])

    r2 = problem1(r, a3)

    sol2 = np.array([
        [27, 6, 5],
        [6, 26, 15]
    ])

    return np.all(r == sol) and np.all(r2 == sol2)

def test_prob_2():
    """Test problem 2"""

    r = problem2(a4, a1, a2)

    sol = np.array([
        [13, 22, 31],
        [8, 16, 24]
    ])

    r2 = problem2(a4, a3, a1)

    sol2 = np.array([
        [146, 70, 21],
        [166, 30, 3]
    ])

    return np.all(r == sol) and np.all(r2 == sol2)

def test_prob_3():
    """Test problem 3"""

    r = problem3(a3, a1, a5)

    sol = np.array([
        [33, 11, 1],
        [6, 81, 42]
    ])

    r2 = problem3(a3, a3, np.transpose(a1))

    sol2 = np.array([
        [577, 6, 3],
        [4, 404, 69]
    ])

    return np.all(r == sol) and np.all(r2 == sol2)

def test_prob_4():
    """Test problem 4"""

    r = problem4(a5[:, 0], a6, a5[:, 1])

    sol = np.array([
        551
    ])

    r2 = problem4(a3[:, 0], a4, a3[:, 1])

    sol2 = np.array([
        1762
    ])

    return np.all(r == sol) and np.all(r2 == sol2)

def test_prob_5():
    """Test problem 5"""

    r = problem5(a3)

    sol = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ])

    r2 = problem5(a5)

    sol2 = np.array([
        [0, 0],
        [0, 0],
        [0, 0]
    ])

    return np.all(r == sol) and np.all(r2 == sol2)

def test_prob_6():
    """Test problem 6"""

    r = problem6(a3)

    sol = np.array([
        1., 1.
    ])

    r2 = problem6(a5)

    sol2 = np.array([
        1., 1., 1.
    ])

    return np.all(r == sol) and np.all(r2 == sol2)

def test_prob_7():
    """Test problem 7"""

    r = problem7(a4, 3)

    sol = np.array([
        [9, 3],
        [7, 4]
    ])

    r2 = problem7(a6, 0)

    sol2 = np.array([
        [6, 3, 5],
        [7, 1, 8],
        [1, 1, 1]
    ])
    
    return np.all(r == sol) and np.all(r2 == sol2)

def test_prob_8():
    """Test problem 8"""

    r = problem8(a4, 1, 0)

    sol = 7

    r2 = problem8(a6, 1, 2)

    sol2 = 8
    
    return r == sol and r2 == sol2

def test_prob_9():
    """Test problem 9"""

    r = problem9(a4, 1)

    sol = 8

    r2 = problem9(a6, 0)

    sol2 = 14
    
    return r == sol and r2 == sol2

def test_prob_10():
    """Test problem 10"""

    r = problem10(a4, 1, 3)

    sol = 2

    r2 = problem10(a6, 2, 10)

    sol2 = 5.8
    
    return r == sol and r2 == sol2

def test_prob_11():
    """Test problem 11"""

    r = problem11(a4, 1)

    sol = np.array([
        [.7408145],
        [.67170966]
    ])

    r2 = problem11(a6, 2)

    sol2 = np.array([
        [-0.70861375, -0.71974195],
        [-0.68834504,  0.34407613],
        [-0.15507307,  0.60297855],
    ])
    
    return np.allclose(r, sol) and np.allclose(r2, sol2)

def test_prob_12():
    """Test problem 12"""

    r = problem12(a4, a4[:, 1])

    sol = np.array([
        [0],
        [1]
    ])

    r2 = problem12(a6, a6[:, 0])

    sol2 = np.array([
        [1],
        [0],
        [0]
    ])
    
    return np.allclose(r, np.transpose(sol)) and np.allclose(r2, np.transpose(sol2))

def test_prob_13():
    """Test problem 13"""

    r = problem13(a4, a4[1, :])

    sol = np.array([
        0, 1
    ])

    r2 = problem13(a6, a6[0, :])

    sol2 = np.array([
        1, 0, 0
    ])
    
    return np.allclose(r, sol) and np.allclose(r2, sol2)

if __name__ == "__main__":
    main()
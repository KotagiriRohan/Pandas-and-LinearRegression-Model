import numpy as np
import pandas as pd
import statsmodels.api as sm

# Part A 1
a1 = 2**5
print("Part A1 )", a1)
print("\n")


# Part A 2
vectora2 = np.array((23, 27, 25, 45, 67))
print("Part A2 )", vectora2)
print("\n")


# Part A 3
vectora3 = pd.Series(
    vectora2, index=["monday", "tuesday", "wednesday", "thursday", "friday"])
print("Part A3 )")
print(vectora3)
print("\n")


# Part A 4
print("Part A4 )", vectora2.mean())
print("\n")


# Part A 5
print("Part A5 )", vectora3.idxmax())
print("\n")


# Part B 1


def functionb1(name):
    return "Hello "+name


print("Part B1 )", functionb1("rohan"))
print("\n")


# Part B 2


def functionb2(num, vec=[]):
    return vec.count(num)


print("Part B2 ) Count of 5: ", functionb2(5, [1, 5, 4, 5, 3, 5, 3]))
print("\n")


# Part B 3


def functionb3(num1, num2, num3):
    sum = 0
    if (num1 % 3):
        sum += num1
    if (num2 % 3):
        sum += num2
    if (num3 % 3):
        sum += num3
    return sum


print("Part B3 )", functionb3(3, 2, 4))
print("\n")


# Part B 4


def functionb4(num):
    flag = False
    if num > 1:
        flag = True
        for i in range(2, num):
            if (num % i) == 0:
                flag = False
                break
    return flag


print("Part B4 )", functionb4(5))
print("\n")
# Part C 1

A = np.array((1, 2, 3))
B = np.array((4, 5, 6))

C = np.row_stack((A, B))
print("Part C 1)\n", C)
print("\n")


# Part C 2
matrixc2 = np.arange(1, 26, dtype=np.int32)
mat2 = matrixc2.reshape((5, 5))

print("Part C 2 )\n", mat2)
print("\n")


# Part C 3
print("Part C 3)\n", mat2[1:3, 1:3])
print("\n")


# Part D 1

df = sm.datasets.get_rdataset('mtcars').data
print("Part D 1)\n", df.head(6))
print("\n")


# Part D 2

print("Part D 2)", df['mpg'].mean())
print("\n")


# Part D 3
print("Part D 3 )\n", df[df['cyl'] == 6].head())
print("\n")


# Part D 4

print("Part D 4)\n", df[["am", "gear", "carb"]].head())
print("\n")


# Part D 5
df["performance"] = df['hp']/df['wt']
print("Part D 5)\n", df['performance'].head())
print("\n")


# Part D 6
print("Part D 6)\n", df[df.index == "Hornet Sportabout"]['mpg'])

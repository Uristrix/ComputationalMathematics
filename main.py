import base64
import io
import pandas as pd

PI = 3.14159265358979323846
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


########################################################################################################################
def reduction(x):
    while x > (2 * PI):
        x -= (2 * PI)
    while x < -(2 * PI):
        x += (2 * PI)
    return x


########################################################################################################################
def cos(x):
    if abs(x) > (2 * PI):
        x = reduction(x)
    s, n, q = 0, 0, 1
    while abs(q) > 0.000001:
        s += q
        q *= -x * x / (2 * n + 2) / (2 * n + 1)
        n = n + 1
    return s, n


########################################################################################################################
def sin(x):
    if x == PI:
        return 0, 0
    if abs(x) > (2 * PI):
        x = reduction(x)
    s, n, q = 0, 1, x
    while abs(q) > 0.000001:
        s += q
        q *= -x * x / (2 * n) / (2 * n + 1)
        n = n + 1
    return s, n


########################################################################################################################
def tg(x):
    s, n1 = sin(x)
    c, n2 = cos(x)
    return (s / c), n1 + n2


########################################################################################################################
def ctg(x):
    s, n1 = sin(x)
    c, n2 = cos(x)
    return (c / s), n1 + n2


########################################################################################################################
def exp(x):
    s, n, q = 1, 1, x
    while abs(q) > 0.000001:
        s += q
        q *= x / (n + 1)
        n = n + 1
    return s, n


########################################################################################################################
def sqrt(x):  # Алгоритм Ньютона
    s, temp, n = x, x, 0
    if x <= 0:
        return 0

    while True:
        temp = (x / temp + temp) / 2
        n = n + 1
        if s > temp:
            s = temp
        else:
            return s


########################################################################################################################
def inf():
    print("Введите X")
    a = float(input())
    print("Какую функцию от X вы хотите найти? ")
    print("\n1)sin(x)", "\n2)cos(x)", "\n3)tg(x)", "\n4)ctg(x)",
          "\n5)sqrt(x)", "\n6)exp(x)", "\n7)Новое значение", "\n8)Выход")
    return a


########################################################################################################################
def calculator(a):
    while True:
        calc = input()
        if calc == '1':
            print(sin(a))
        if calc == '2':
            print(cos(a))
        if calc == '3':
            print(tg(a))
        if calc == '4':
            print(ctg(a))
        if calc == '5':
            print(sqrt(a))
        if calc == '6':
            print(exp(a))
        if calc == '7':
            a = inf()
        if calc == '8':
            exit()


# calculator(inf())
# print("Введите X")
# a = float(input())
# print("Введите B")
# b = float(input())
########################################################################################################################
########################################################################################################################
def parse(s):
    delims = ["+", "-", "*", "^", "/", "(", ")", "cos", "sin", "tg", "ctg",
              "sqrt", "exp", "=", ";", "y'", "y''", "y'''", "y''''", "pi"]
    lex = []
    tmp = ""
    for a in s:
        if a != " ":
            if a in delims:
                if tmp != "":
                    lex += [tmp]
                lex += [a]
                tmp = ""
            else:
                tmp += a
    if tmp != "":
        lex += [tmp]
    return lex


########################################################################################################################
def prty(o):
    if o == "+" or o == "-":
        return 1
    elif o == "sin" or o == "cos" or o == "tg" or o == "ctg" or o == "exp":
        return 3
    elif o == "*" or o == "/":
        return 2
    elif o == "^" or o == 'sqrt':
        return 3
    elif o == "(":
        return 0


########################################################################################################################
def rpn(s):
    lex = parse(s)
    s2 = []
    r = []
    oper = ["+", "-", "*", "/", "(", ")", "^", "sin", "cos", "tg", "ctg", "sqrt", "exp"]
    for a in lex:
        if a == "(":
            s2 = [a] + s2
        elif a in oper:
            if s2 == []:
                s2 = [a]
            elif a == ")":
                while True:
                    q = s2[0]
                    s2 = s2[1:]
                    if q == "(":
                        break
                    r += [q]
            elif prty(s2[0]) < prty(a):
                s2 = [a] + s2
            else:
                while True:
                    if s2 == []:
                        break
                    q = s2[0]
                    r += [q]
                    s2 = s2[1:]
                    if prty(q) == prty(a):
                        break
                s2 = [a] + s2
        else:
            r += [a]
    while (s2 != []):
        q = s2[0]
        r += [q]
        s2 = s2[1:]
    return r


# str3 = "2*tg(x+1)/sin(x)^2"
# print(rpn("2*tg(x+1)/sin(x)^2"))
########################################################################################################################
def replace(x, rpn):
    copy = rpn[:]
    while 'x' in copy:
        copy[copy.index('x')] = x
    return copy


########################################################################################################################
def replace_dif(x, y, rpn):
    copy = replace(x, rpn)
    for key in y:
        while key in copy:
            copy[copy.index(key)] = y[key]
    print(copy)
    return copy


########################################################################################################################
def replace_y_to_x(rpn):
    copy = rpn[:]
    while 'y' in copy:
        copy[copy.index('y')] = 'x'

    return copy


########################################################################################################################
def Operation(oper, num):
    if oper == 'sin':
        res, n = sin(float(num))
    if oper == 'cos':
        res, n = cos(float(num))
    if oper == 'tg':
        res, n = tg(float(num))
    if oper == 'ctg':
        res, n = ctg(float(num))
    if oper == 'sqrt':
        res, n = sqrt(float(num))
    if oper == 'exp':
        res, n = exp(float(num))
    return res, n


########################################################################################################################
def Operation2(oper, num1, num2):
    if oper == '+':
        return float(num1) + float(num2)
    if oper == '-':
        return float(num1) - float(num2)
    if oper == '*':
        return float(num1) * float(num2)
    if oper == '/':
        return float(num1) / float(num2)
    if oper == '^':
        return float(num1) ** float(num2)


########################################################################################################################
########################################################################################################################
def Functions(equa, x, y, how):
    if how == "eight":
        equa = replace_pi(equa)
        equa = replace(x, equa)
    if how == "ten":
        equa = replace_y_to_x(equa)
        equa = replace(x, equa)
    if how == "sixth":
        equa = replace_dif(x, y, equa)
    else:
        equa = replace_pi(equa)
        equa = replace(x, equa)
    delims = ["+", "-", "*", "^", "/"]
    delims2 = ["sin", "cos", "tg", "ctg", "sqrt", "exp"]
    n, temp = 0, 0
    while len(equa) > 1:
        for i in equa:
            if i in delims2:
                first = equa.pop(equa.index(i) - 1)
                equa[equa.index(i)], temp = Operation(equa[equa.index(i)], first)
                n += temp
                # equa[equa.index(i)] = str(equa[equa.index(i)])
                break
            if i in delims:
                second = equa.pop(equa.index(i) - 1)
                first = equa.pop(equa.index(i) - 1)
                equa[equa.index(i)] = str(Operation2(equa[equa.index(i)], first, second))
                break
    if how == "first":
        return float(equa[0]), n
    if how == "third" or how == "fourth" or how == "sixth" or how == "eight" or how == "ten":
        return float(equa[0])


########################################################################################################################
def derivative(lst, der):
    rpn = lst.copy()
    i = 0
    if der == 'x':
        c = 'y'
    else:
        c = 'x'
    while i < len(rpn):
        if rpn[i] == der:
            if rpn[i - 1].isdigit() and rpn[i + 1].isdigit():
                rpn[i - 1] = str(float(rpn[i - 1]) * float(rpn[i + 1]))

                rpn[i + 1] = str(float(rpn[i + 1]) - 1)
                if rpn[i + 1] == '1.0':
                    rpn.pop(i + 1)
                    rpn.pop(i + 1)

            elif (i - 1 == -1 or not rpn[i - 1].isdigit()) and rpn[i + 1].isdigit():
                koef = rpn[i + 1]
                rpn[i + 1] = str(float(rpn[i + 1]) - 1)
                if rpn[i + 1] == '1.0':
                    rpn.pop(i + 1)
                    # rpn.pop(i + 1)
                rpn.insert(i, str(float(koef)))
                i += 1
                if rpn[i + 1] == '^':
                    rpn[i + 1] = '*'
                else:
                    rpn.insert(i + 3, '*')

            elif not rpn[i + 1].isdigit():
                rpn.pop(i)
                rpn.pop(i)

        elif rpn[i].isdigit() and (rpn[i - 1] == '-' or rpn[i - 1] == '+') and (rpn[i + 1] == '-' or rpn[i + 1] == '+'):
            rpn.pop(i)
            rpn.pop(i)

        elif rpn[i] == c:
            flag = False
            if rpn[i - 1].isdigit():
                rpn.pop(i - 1)
                i -= 1
                flag = True
            if rpn[i + 1].isdigit():
                rpn.pop(i + 1)
                rpn.pop(i + 1)
            rpn.pop(i)
            rpn.pop(len(rpn) - 1)
            if flag == True:
                rpn.pop(i)
            i -= 1
        i += 1
    return rpn


stri = "2*x^2+5*y^2"


# print(rpn(stri))
# print(derivative(rpn(stri), 'x'))
########################################################################################################################
def graf(func, a, b):
    x, y, a, b = [], [], float(a), float(b)
    while a < b:
        x.append(float(a))
        y.append(float(Functions(func, a, 0, "third")))
        a += 0.1
    return x, y


########################################################################################################################
def check(func, a, b):
    a, b = float(a), float(b)
    if Functions(func, a, 0, "third") * Functions(func, b, 0, "third") > 0:
        return "Error: no roots"
    if Functions(func, a, 0, "third") * Functions(func, b, 0, "third") == 0:
        if Functions(func, a, 0, "third") == 0:
            return a
        else:
            return b


########################################################################################################################

def gap(func):
    a1, b1 = -0.5, 0
    a2, b2 = 0, 0.5
    while True:
        if Functions(func, a1, 0, "third") * Functions(func, b1, 0, "third") <= 0:
            return a1, b1
        if Functions(func, a2, 0, "third") * Functions(func, b2, 0, "third") <= 0:
            return a2, b2
        b1 = a1
        a1 -= 0.5

        a2 = b2
        b2 += 0.5


########################################################################################################################
def MethodPosFind(func):
    # a, b = float(a), float(b)
    a, b = gap(func)
    x = a + 0.000001
    temp = Functions(func, a, 0, "third")
    while Functions(func, x, 0, "third") * temp > 0:
        x += 0.000001
    return x


########################################################################################################################
def Dichot(func):  # Метод половиного деления
    # a, b = float(a), float(b)
    a, b = gap(func)
    while b - a > 0.000001:
        c = (a + b) / 2
        if Functions(func, c, 0, "third") == 0:
            return c
        if Functions(func, b, 0, "third") * Functions(func, c, 0, "third") < 0:
            a = c
        else:
            b = c
    return (a + b) / 2


########################################################################################################################
def Hord(func):  # Метод хорд
    a, b = gap(func)
    while abs(b - a) > 0.000001:
        a = b - (b - a) * Functions(func, b, 0, "third") / (
                Functions(func, b, 0, "third") - Functions(func, a, 0, "third"))
        b = a - (a - b) * Functions(func, a, 0, "third") / (
                Functions(func, a, 0, "third") - Functions(func, b, 0, "third"))
    return b


########################################################################################################################
def Nuton(func):
    a, b = gap(func)
    while abs(b - a) > 0.000001:
        c = b
        b = a - (Functions(func, a, 0, "third") / (Functions(derivative(func, 'x'), a, 0, "third")))
        a = c
    return b


# print(rpn("5*x^3-8*x^2-8*x+5"))
########################################################################################################################
# Integral
def rectangle_right(func, a, b):
    a, b, res, res_pred = float(a), float(b), 0, 1
    n = 1000

    while (abs(res_pred - res)) > 0.000001:

        res_pred = res
        res = 0
        n *= 2
        h = (b - a) / n
        temp = a + h

        for i in range(n):
            res += Functions(func, temp, 0, "fourth")
            temp += h
        res *= h

    return res


########################################################################################################################
def rectangle_left(func, a, b):
    a, b, res, res_pred = float(a), float(b), 0, 1
    n = 1000

    while (abs(res_pred - res)) > 0.000001:

        res_pred = res
        res = 0
        n *= 2
        h = (b - a) / n

        temp = a
        for i in range(n):
            res += Functions(func, temp, 0, "fourth")
            temp += h
        res *= h

    return res


########################################################################################################################
def trap(func, a, b):
    a, b, res, res_pred = float(a), float(b), 0, 1
    n = 1000

    while (abs(res_pred - res)) > 0.000001:

        res_pred = res
        res = 0
        n *= 2
        h = (b - a) / n
        temp = a

        for i in range(n):
            res += (Functions(func, temp, 0, "fourth") + Functions(func, temp + h, 0, "fourth")) / 2
            temp += h
        res *= h

    return res


########################################################################################################################
def simp(func, a, b):
    # a, b, res = float(a), float(b), 0
    # h = 0.000001 ** 0.25
    # n = (b - a) / h
    # if round(n) < n:
    #     n += 1
    # n = round(n)
    # while n % 4 != 0:
    #     n += 1
    # h = (b - a) / 10000000
    # temp = a
    # print(n, h)
    # #n = 10000000
    # h = (b - a) / n
    a, b, res, res_pred = float(a), float(b), 0, 1
    n = 1000

    while (abs(res_pred - res)) > 0.000001:
        res_pred = res
        res = 0
        n *= 2
        h = (b - a) / n
        temp = a
        for i in range(n):
            if i == 0 or i == n - 1:
                res += Functions(func, temp, 0, "fourth")
            elif i % 2 != 0:
                res += 4 * Functions(func, temp, 0, "fourth")
            else:
                res += 2 * Functions(func, temp, 0, "fourth")
            temp += h
        res *= (h / 3)
    return res


########################################################################################################################
def create_dict(str_):
    res = dict()
    str_ = parse(str_)
    for i in range(len(str_)):
        if str_[i] == "=":
            res[str_[i - 1]] = str_[i + 1]
    return res


########################################################################################################################
def create_column(x_arr, y_arr):
    x_arr = {'x': x_arr}
    x_arr.update(y_arr)
    return [{'name': '{}'.format(key), 'id': '{}'.format(key)} for key in x_arr]


########################################################################################################################
def create_data(x_arr, y_arr):
    x_arr = {'x': x_arr}
    x_arr.update(y_arr)
    for key in x_arr:
        for i in range(len(x_arr[key])):
            x_arr[key][i] = float(x_arr[key][i])
            if round(x_arr[key][i], 6) < x_arr[key][i]:
                x_arr[key][i] += 0.0000001
            x_arr[key][i] = round(x_arr[key][i], 6)
    return [{'{}'.format(key): x_arr[key][j] for key in x_arr} for j in range(len(x_arr['x']))]


# print(simp("x^2*sin(x)", '0', '1'))
########################################################################################################################
def Eyler(func, x, y_arr):
    x, x_arr, h = float(x), [float(x)], 0.1
    for key in y_arr:
        y_arr[key] = [float(y_arr[key])]
    for i in range(1, 10):
        x += h
        x_arr.append(x)
        for key in y_arr:
            y_temp = y_arr.copy()
            for t_key in y_temp:
                y_temp[t_key] = y_arr[t_key][i - 1]
            y_arr[key].append(y_arr[key][i - 1] + Functions(func, x_arr[i - 1], y_temp, "sixth") * h)
    return x_arr, y_arr


########################################################################################################################
def Runge(func, x, y_arr):
    x, x_arr, h = float(x), [float(x)], 0.1
    for key in y_arr:
        y_arr[key] = [float(y_arr[key])]
    for i in range(1, 11):
        x += h
        x_arr.append(x)
        for key in y_arr:
            y_temp = y_arr.copy()
            for t_key in y_temp:
                y_temp[t_key] = y_arr[t_key][i - 1]
            k1 = Functions(func, x_arr[i - 1], y_temp, "sixth")
            y_temp[key] += h * k1 / 2
            k2 = Functions(func, x_arr[i - 1] + h / 2, y_temp, "sixth")
            y_temp[key] -= (h * k1 / 2)
            y_temp[key] += h * k2 / 2
            k3 = Functions(func, x_arr[i - 1] + h / 2, y_temp, "sixth")
            y_temp[key] -= h * k2 / 2
            y_temp[key] += h * k3
            k4 = Functions(func, float(x_arr[i - 1]) + h, y_temp, "sixth")
            y_arr[key].append(y_arr[key][i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    print(x_arr, y_arr)
    return x_arr, y_arr


########################################################################################################################
def parse_num(str):
    temp = ''
    res = []
    for s in str:
        if s == ' ':
            res.append(float(temp))
            temp = ''
        temp += s
    res.append(float(temp))
    return res


########################################################################################################################


def Lagranj(x_arr, y_arr, x):
    L = 0
    for i in range(len(x_arr)):
        temp = y_arr[i]
        for j in range(len(x_arr)):
            if i != j:
                temp *= (x - x_arr[j]) / (x_arr[i] - x_arr[j])
        L += temp
    return L


########################################################################################################################
def interpolations(x_arr, y_arr):
    x_new, y_new = [], []
    h = 0.01
    temp = float(int(x_arr[0]))
    while temp < x_arr[len(x_arr) - 1]:
        x_new.append(temp)
        y_new.append(Lagranj(x_arr, y_arr, temp))
        temp += h
    return x_new, y_new


########################################################################################################################

def least_square(x_arr, y_arr):
    xy, x2 = 0, 0
    for i in range(len(x_arr)):
        xy += x_arr[i] * y_arr[i]
        x2 += x_arr[i] ** 2

    a = ((len(x_arr)) * xy - (sum(x_arr) * sum(y_arr))) / ((len(x_arr)) * x2 - sum(x_arr) ** 2)
    b = (sum(y_arr) - a * sum(x_arr)) / (len(x_arr))

    return str(a) + "*x+" + str(b)


########################################################################################################################
def matrix_koef(koef, rpn):
    print(koef, rpn)
    for i in range(len(rpn)):
        if rpn[i] == 'x':
            rpn[i] = koef.pop(0)
    print(koef, rpn)
    return rpn


########################################################################################################################
import numpy as np


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    df = None
    arr = []
    y = []
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';')
    elif 'xls' in filename:
        df = pd.read_excel(io.BytesIO(decoded), sep=';')
    df = df.values

    for i in range(len(df)):
        arr.append([])
        y.append(df[i][len(df[i]) - 1])
        for j in range(len(df[i]) - 1):
            arr[i].append(df[i][j])

    return arr, y


########################################################################################################################
def Gauss(a, b):
    res = np.zeros(len(a))

    for i in range(len(a)):
        for j in range(i + 1, len(a)):

            temp = a[j][i] / a[i][i]

            for k in range(i, len(a)):
                a[j][k] -= temp * a[i][k]

            b[j] -= temp * b[i]

    for i in range(len(a) - 1, -1, -1):
        temp2 = 0
        for j in range(i + 1, len(a)):
            temp2 += a[i][j] * res[j]

        res[i] = (b[i] - temp2) / a[i][i]
    return res


a = [[1, 2, 3],
     [3, 5, 7],
     [1, 3, 4]]
b = [3, 0, 1]


# print(Gauss(a, b))
# print(np.linalg.solve(a, b))
########################################################################################################################
def matrix_least_square(z_arr, y_arr):
    y_np = np.zeros(len(z_arr[0]) + 1)
    z_np = np.zeros((len(z_arr[0]) + 1) ** 2).reshape((len(z_arr[0]) + 1), (len(z_arr[0]) + 1))

    N = len(y_arr)
    y_np[0] = sum(y_arr)
    z_np[0][0] = N

    for i in range(1, len(y_np)):
        for j in range(N):
            y_np[i] += y_arr[j] * z_arr[j][i - 1]

    # print(y_np)
    for i in range(len(z_np)):
        for j in range(len(z_np)):
            if i == 0 and j == 0:
                continue

            for k in range(N):
                if j == 0:
                    z_np[i][j] += z_arr[k][i - 1]
                elif i == 0:
                    z_np[i][j] += z_arr[k][j - 1]
                else:
                    z_np[i][j] += z_arr[k][j - 1] * z_arr[k][i - 1]
    # print(z_np)
    # print(np.linalg.solve(z_np, y_np))
    return list(np.linalg.solve(z_np, y_np))


########################################################################################################################
def create_equation(equa):
    print(equa)
    str_equa = ''
    for i in range(len(equa)):
        if i == 0 and str(equa[i])[0] == '-':
            str_equa += '0'
        if str(equa[i])[0] != '-':
            str_equa += '+'
        str_equa += str(equa[i])
        if i != 0:
            str_equa += "*x"
    print(str_equa)
    return str_equa


########################################################################################################################\


z_arr = [[5, 7, 12, 2, 3],
         [6, 3, 5, 9, 1],
         [5, 1, 4, 6, 7]]

y_arr = [5, 7, 8]


# print(matrix_least_square(z_arr, y_arr))
# print(np.linalg.solve(np.array(z_arr), np.array(y_arr)))
# print(np.linalg.solve(np.array(z_arr), np.array(y_arr)))

########################################################################################################################
def Furry(N, M):
    a, b = 1, 1
    dx = 1 / (N - 1)
    dy = b / a / (M - 1)

    temp = []
    for i in range(N):
        temp.append([0] * M)

    for j in range(N):
        temp[0][j] = 1
        temp[N - 1][j] = 1
    for i in temp:
        print(i)

    if dx < dy:
        dt = dx * dx / 2
    else:
        dt = dy * dy / 2
    dt *= 0.9
    while True:
        Err = 0
        for i in range(1, N - 1):
            for j in range(1, M - 1):
                T1 = temp[i][j] + dt * ((temp[i + 1][j] - 2 * temp[i][j] + temp[i - 1][j]) / (dx ** 2) +
                                        (temp[i][j + 1] - 2 * temp[i][j] + temp[i][j - 1]) / (dy ** 2))

                Err += abs(T1 - temp[i][j]) / dt
                temp[i][j] = T1

        if Err < 0.1:
            break
    return temp


def replace_pi(equa):
    copy = equa[:]
    while 'pi' in copy:
        copy[copy.index('pi')] = str(PI)
    return copy


########################################################################################################################
def My_Furry(N, M, func, a):
    temp = []
    for i in range(N + 1):
        temp.append([0] * (M + 1))

    h = 1 / M
    ht = 60
    for i in range(M + 1):
        temp[0][i] = Functions(func, h * i, 0, "eight")

    for i in range(N + 1):
        temp[i][0] = temp[0][0]
        temp[i][M] = temp[0][M]

    for i in temp:
        print(i)

    for i in range(1, N + 1):
        for j in range(1, M):
            print(a * ht / (h * h))
            temp[i][j] = temp[i - 1][j] + (a * a * ht / (h * h)) * (
                        temp[i - 1][j + 1] - 2 * temp[i - 1][j] + temp[i - 1][j - 1])

    # for i in temp:
    #     print(i)

    return temp


########################################################################################################################

def gradient_descent(func, X, Y, d):
    X_der = Functions(derivative(func, 'x'), X, 0, "ten")
    Y_der = Functions(derivative(func, 'y'), Y, 0, "ten")

    while ((X_der * d) ** 2 + (Y_der * d) ** 2) ** 0.5 > 0.000001:
        print(X)
        X = X - X_der * d
        Y = Y - Y_der * d
        X_der = Functions(derivative(func, 'x'), X, 0, "ten")
        Y_der = Functions(derivative(func, 'y'), Y, 0, "ten")

    return X - X_der * d, Y - Y_der * d


# print(gradient_descent('x^2+y^2', 1, 1, 0.1))
########################################################################################################################

########################################################################################################################
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)
app.title = "Camputational Mathematics"
app.layout = html.Div(

    # style={'background-image': 'url("https://a.rihaansfics.com/wp-content/uploads/2017/12/grey_gradient.jpg")'},
    # style={'background-image': 'url("https://img.wallpapersafari.com/desktop/1920/1080/14/63/UXBz7N.jpg")'},'background-repeat': 'no-repeat'
    # style={'background': 'radial-gradient(circle, rgba(185,241,252,1) 45%, rgba(255,255,255,1) 79%',
    #      'background-size': '50%', },

    # 'radial-gradient(circle, rgba(185,241,252,1) 45%, rgba(255,255,255,1) 79%, rgba(255,255,255,1 89%)'
    children=[
        html.Div(
            [
                # html.Audio(src="assets/lit.ogg"),
                html.Div('Camputational Mathematics', className="app-header-title"),
                # lab1
                html.Div("1 Laboratory work", className="app-lab-title")
            ], style={'textAlign': 'center', "padding": "30px"}),

        html.Div(
            [
                html.Div("   Введите уравнение:", className="lab_info"),
                html.Div([" Input: ", dcc.Input(id='func', value='exp(cos(x^2))*sin(x)', type='text')]),
                html.Div("   Введите X:", className="lab_info"),
                html.Div([" Input: ", dcc.Input(id='X', value='2.124', type='text')]),
                html.Br(),
                html.Button(id='button', n_clicks=0, children='Submit'),
                html.Div(id='output'),

            ], style={'width': '30%', 'display': 'inline-block', 'float': 'center', }, className='backside'
        ),

        html.Div([
            html.Div(id="lolkek")
        ],  # style={'width': '57%', 'float': 'right', 'display': 'inline-block', 'padding': '-800px 200px 0px 100px'}
        ),
        html.Div("                                                          ", style={"padding": "30px"}),
        # lab3
        html.Div([
            html.Div("3 Laboratory work", className="app-lab-title"),
            # html.Video(src="https://www.youtube.com/watch?v=hXQxSi34GWY"),
        ], style={'textAlign': 'center'}),
        html.Div(
            [
                html.Div(
                    dcc.Dropdown(
                        id="dropdown",
                        options=[
                            {'label': 'Метод Хорд (секущих)', 'value': 'Hord'},
                            {'label': 'Метод Ньютона', 'value': 'Nuton'},
                            {'label': 'Метод Половинного деления', 'value': 'Dichot'},
                            {'label': 'Метод Последовательного поиска (НЕ ВЫБИРАТЬ!!! ДОЛГО СЧИТАЕТ!!!)',
                             'value': 'Pos'}
                        ], style={"padding": "0px 0px 0px 0px"}
                    ),
                ),
                html.Div("    ", style={"padding": "20px"}),
                html.Div([
                    html.Div("   Введите уравнение:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='func2', value='5*x^3-8*x^2-8*x+5', type='text')]),
                    # html.Div("   Введите A:", className="lab_info"),
                    # html.Div([" Input: ", dcc.Input(id='A', value='-5', type='text')]),
                    # html.Div("   Введите B:", className="lab_info"),
                    # html.Div([" Input: ", dcc.Input(id='B', value='5', type='text')]),
                    html.Br(),
                    html.Button(id='button2', n_clicks=0, children='Submit'),
                    html.Div(id='output2')
                ], className='backside', style={"padding": "30px 40px 40px 40px"}
                )

            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px 0px 10px 10px'}
        ),

        html.Div([
            dcc.Graph(id="graph")
        ], style={'width': '57%', 'float': 'right', 'display': 'inline-block', 'padding': '-800px 200px 0px 100px'}
        ),

        html.Div([
            html.Div("4 Laboratory work", className="app-lab-title"),
        ], style={'textAlign': 'center', 'padding': '170px 0px 0px 0px'}),

        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown2",
                    options=[
                        {'label': 'Метод прямоугольника(левая ступенька)', 'value': 'rectL'},
                        {'label': 'Метод прямоугольника(правая ступенька)', 'value': 'rectR'},
                        {'label': 'Метод трапеций', 'value': 'trap'},
                        {'label': 'Метод Симпсона', 'value': 'Simp'}
                    ]
                ),
                html.Div("    ", style={"padding": "20px"}),
                html.Div([
                    html.Div("   Введите уравнение:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='func3', value='x^2*sin(x)', type='text')]),
                    html.Div("   Введите A:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='A2', value='0', type='text')]),
                    html.Div("   Введите B:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='B2', value='1', type='text')]),
                    html.Br(),
                    html.Button(id='button3', n_clicks=0, children='Submit'),
                    html.Div(id='output3')
                ], className='backside'
                ),

            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px 0px 0px 0px'}
        ),

        html.Div([
            html.Div("5 Laboratory work", className="app-lab-title"),
        ], style={'textAlign': 'center', 'padding': '70px 0px 0px 0px'}),

        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown5",
                    options=[
                        {'label': 'Метод Эйлера', 'value': 'Eyler'},
                        {'label': 'Метод Рунге - Кутта', 'value': 'Runge'},
                    ]
                ),
                html.Div("    ", style={"padding": "20px"}),
                html.Div([
                    html.Div("   Введите уравнение:", className="lab_info"),
                    html.Div([" Input: y''=", dcc.Input(id='func5', value='x^2-2*y', type='text')]),
                    html.Div("   Введите начальные условия:", className="lab_info"),
                    html.Div([" Input: x, y, y'....=", dcc.Input(id='koef', value='x=0; y=1;', type='text')]),
                    html.Div("    ", style={"padding": "20px"}),
                    html.Button(id='button5', n_clicks=0, children='Submit'),
                ], className='backside'),
                html.Div("    ", style={"padding": "20px"}),
                html.Div([table.DataTable(id='table')])
            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px 0px 10px 10px'}
        ),
        html.Div([
            dcc.Graph(id="graph5")
        ], style={'width': '57%', 'float': 'right', 'display': 'inline-block', 'padding': '-800px 0px 0px 0px'}
        ),

        html.Div([
            html.Div("6 Laboratory work", className="app-lab-title"),
        ], style={'textAlign': 'center', 'padding': '140px 0px 0px 0px'}),

        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown6",
                    options=[
                        {'label': 'Интерполяционный многочлен Лагранжа', 'value': 'Lag'},
                        {'label': 'Метод наименьших квадратов', 'value': 'Sq'},
                        {'label': 'Метод наименьших квадратов(матрица)', 'value': 'Matrix'}
                    ]
                ),
                html.Div("    ", style={"padding": "20px"}),
                html.Div([
                    html.Div("   Введите значения x:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='X6', value='0 1 2 3 4 5', type='text')]),
                    html.Div("   Введите значения y:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='Y6', value='86 85 86 84 88 86', type='text')]),
                    html.Div("   Введите значения для наименьших квадратов:", className="lab_info"),
                    html.Div([" Input:", dcc.Input(id='func6', value='5 7 12 4 7', type='text')]),
                    html.Div([dcc.Upload(html.Button('Upload File'), id='upload')]),

                    html.Br(),
                    html.Button(id='button6', n_clicks=0, children='Submit'),
                    html.Div(id='output6')
                ], className='backside'
                ),

            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px 0px 0px 0px'}
        ),
        html.Div([
            dcc.Graph(id="graph6")
        ], style={'width': '57%', 'float': 'right', 'display': 'inline-block', 'padding': '-800px 0px 0px 0px'}
        ),
        html.Div([
            html.Div("8 Laboratory work", className="app-lab-title"),
        ], style={'textAlign': 'center', 'padding': '140px 0px 0px 0px'}),

        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown8",
                    options=[
                        {'label': 'Краевая задача с пластиной', 'value': 'plate'},
                        {'label': 'Краевая задача со стержнем', 'value': 'kernel'},
                    ]
                ),
                html.Div("    ", style={"padding": "20px"}),
                html.Div([
                    html.Div("   Введите значения M:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='M', value='100', type='text')]),
                    html.Div("   Введите значения N:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='N', value='100', type='text')]),
                    html.Div("    ", style={"padding": "20px"}),
                    html.Button(id='button8', n_clicks=0, children='Submit'),
                ], className='backside'
                ),

            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px 0px 0px 0px'}
        ),
        html.Div([
            dcc.Graph(id="graph8")
        ], style={'width': '57%', 'float': 'right', 'display': 'inline-block', 'padding': '-800px 0px 0px 0px'}
        ),
        html.Div([
            html.Div("10 Laboratory work", className="app-lab-title"),
        ], style={'textAlign': 'center', 'padding': '140px 0px 0px 0px'}),
        html.Div(
            [
                html.Div([
                    html.Div("   Введите функцию:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='func10', value='x^2+y^2', type='text')]),
                    html.Div("   Введите значение X:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='X10', value='1', type='text')]),
                    html.Div("   Введите значение Y:", className="lab_info"),
                    html.Div([" Input: ", dcc.Input(id='Y10', value='1', type='text')]),
                    html.Div("    ", style={"padding": "20px"}),
                    html.Button(id='button10', n_clicks=0, children='Submit'),
                    html.Div(id='output10')
                ], className='backside'
                ),

            ], style={'width': '40%', 'display': 'inline-block', 'padding': '10px 0px 0px 0px'}
        ),
    ])

clicks, clicks2, clicks3, clicks5, clicks6, clicks8, clicks10 = 0, 0, 0, 0, 0, 0, 0


@app.callback([Output(component_id='output', component_property='children'),
               Output(component_id='lolkek', component_property='children')],
              [Input("func", "value"), Input("X", "value"), Input('button', 'n_clicks')])
def update_first_lab(func, x, click):
    global clicks
    if not func or not x or click == clicks:
        return 'Output: ', ''
    elif x == '69':
        return 'Output:', html.Div(
            [
                html.Img(
                    src='http://img10.joyreactor.cc/pics/post/full/Anime-Kantai-Collection-Isonami-hangaku-3815554.gif',
                    height='200', width='200'),
                html.Img(
                    src='http://img10.joyreactor.cc/pics/post/full/Anime-Kantai-Collection-Isonami-hangaku-3815554.gif',
                    height='200', width='200'),
                html.Img(src='https://pa1.narvii.com/6900/f4b350aa9a88a2d111d5ed93b1c8c15718c6e1b6r1-540-304_hq.gif',
                         height='200', width='200'),
                html.Img(
                    src='http://img10.joyreactor.cc/pics/post/full/Anime-Kantai-Collection-Isonami-hangaku-3815554.gif',
                    height='200', width='200'),
                html.Img(
                    src='http://img10.joyreactor.cc/pics/post/full/Anime-Kantai-Collection-Isonami-hangaku-3815554.gif',
                    height='200', width='200'),
                html.Audio(src="lit.mp3", autoPlay='AUTOPLAY')
            ]
        )
    else:
        clicks += 1
        return 'Output: {}'.format(str(Functions(rpn(func), x, 0, "first"))), ''


@app.callback([Output(component_id='output2', component_property='children'),
               Output('graph', 'figure')],
              [Input("func2", "value"),
               # Input("A", "value"),
               # Input("B", "value"),
               Input('button2', 'n_clicks'),
               Input('dropdown', 'value')])
def update_third_lab(func, click2, dropdown):
    global clicks2
    fig = go.Figure()
    if not func or click2 == clicks2 or dropdown is None:
        return 'Output: ', fig
    else:
        func = rpn(func)
        a, b = gap(func)
        a = a - 20
        b = b + 20
        x_arr, y_arr = graf(func, a, b)
        fig.add_trace(go.Scatter(x=x_arr, y=y_arr, name="Function"))
        clicks2 += 1
        if dropdown == 'Hord':
            return '    Output: {}'.format(str(Hord(func))), fig
        if dropdown == 'Nuton':
            return '    Output: {}'.format(str(Nuton(func))), fig
        if dropdown == 'Dichot':
            return '    Output: {}'.format(str(Dichot(func))), fig
        if dropdown == 'Pos':
            return '    Output: {}'.format(str(MethodPosFind(func))), fig


@app.callback(Output(component_id='output3', component_property='children'),
              [Input("func3", "value"),
               Input("A2", "value"),
               Input("B2", "value"),
               Input('button3', 'n_clicks'),
               Input('dropdown2', 'value')])
def update_fourth_lab(func, a, b, click3, dropdown2):
    global clicks3
    if not func or not a or not b or click3 == clicks3 or dropdown2 is None:
        return 'Output: '
    else:
        clicks3 += 1
        if dropdown2 == 'rectL':
            return '    Output: {}'.format(str(rectangle_left(rpn(func), a, b)))
        if dropdown2 == 'rectR':
            return '    Output: {}'.format(str(rectangle_right(rpn(func), a, b)))
        if dropdown2 == 'trap':
            return '    Output: {}'.format(str(trap(rpn(func), a, b)))
        if dropdown2 == 'Simp':
            return '    Output: {}'.format(str(simp(rpn(func), a, b)))


@app.callback([Output('table', 'columns'),
               Output('table', 'data'),
               Output("graph5", "figure")],

              [Input("func5", "value"),
               Input('button5', 'n_clicks'),
               Input('dropdown5', 'value'),
               Input('koef', 'value')])
def update_fifth_lab(func, click5, dropdown5, koef):
    global clicks5
    fig, x_arr, y_arr, column, data = go.Figure(), [], [], None, None
    if not func or not koef or click5 == clicks5 or dropdown5 is None:
        return None, None, fig
    else:
        clicks5 += 1
        dictionary = create_dict(koef)
        x = dictionary.pop("x")
        if dropdown5 == 'Eyler':
            x_arr, y_arr = Eyler(rpn(func), x, dictionary)
        if dropdown5 == "Runge":
            x_arr, y_arr = Runge(rpn(func), x, dictionary)
        column = create_column(x_arr, y_arr)
        data = create_data(x_arr, y_arr)
        fig.add_trace(go.Scatter(x=x_arr, y=y_arr['y'], name="Differencial"))
        # else:
        #    print(len(x_arr), len(y_arr['y']), len(y_arr["y'"]))
        #    fig.add_trace(go.Surface(x=x_arr, y=list(y_arr["y"]), z=list(y_arr["y'"]), name="Differencial"))
        return column, data, fig


@app.callback([Output("graph6", "figure"),
               Output(component_id='output6', component_property='children')],
              [Input('button6', 'n_clicks'),
               Input('dropdown6', 'value'),
               Input('func6', 'value'),
               Input('X6', 'value'),
               Input('Y6', 'value'),
               Input('upload', 'contents')],
              State('upload', 'filename'),
              State('upload', 'last_modified')
              )
def update_sixth_lab(click6, dropdown6, func, X6, Y6, contents, filename, last_modified):
    global clicks6
    fig, x_arr, y_arr = go.Figure(), [], []

    if not X6 or not Y6 or click6 == clicks6 or dropdown6 is None:
        return fig, None

    else:
        clicks6 += 1
        X6 = parse_num(X6)
        Y6 = parse_num(Y6)

        if dropdown6 == 'Lag':
            fig.add_trace(go.Scatter(x=X6, y=Y6, mode='markers'))
            x_arr, y_arr = interpolations(X6, Y6)
            fig.add_trace(go.Scatter(x=x_arr, y=y_arr, name="Lagranj"))
            return fig, None

        if dropdown6 == 'Sq':
            fig.add_trace(go.Scatter(x=X6, y=Y6, mode='markers'))
            x_arr, y_arr = graf(rpn(least_square(X6, Y6)), X6[0], X6[len(X6) - 1])
            fig.add_trace(go.Scatter(x=x_arr, y=y_arr, name="Square"))
            return fig, None

        if dropdown6 == 'Matrix' and contents is not None:
            arr, y = parse_contents(contents, filename)
            polsk = rpn(create_equation(matrix_least_square(arr, y)))
            return fig, Functions(matrix_koef(parse_num(func), polsk), 0, 0, "third")


@app.callback(Output("graph8", "figure"),
              [Input('dropdown8', 'value'),
               Input('button8', 'n_clicks'),
               Input('M', 'value'),
               Input('N', 'value')])
def update_eight_lab(dropdown8, click8, M, N):
    global clicks8
    fig, x_arr, y_arr = go.Figure(), [], []

    if not M or not N or click8 == clicks8 or dropdown8 is None:
        return fig
    else:
        clicks8 += 1

        if dropdown8 == 'plate':
            temp = Furry(int(N), int(M))
            fig.add_trace(go.Contour(z=temp))

        if dropdown8 == 'kernel':
            temp = My_Furry(int(N), int(M), rpn("sin(pi*x)"), (2.3 * 0.00001))
            fig.add_trace(go.Contour(z=temp))

        return fig


@app.callback(Output(component_id='output10', component_property='children'),
              [Input("func10", "value"),
               Input('X10', 'value'),
               Input('Y10', 'value'),
               Input('button10', 'n_clicks')])
def update_ten_lab(func, X, Y, click10):
    global clicks10
    if not X or not Y or click10 == clicks10:
        return "Output:"
    else:
        return 'Output: {}'.format(str(gradient_descent(rpn(func), float(X), float(Y), 0.1)))


if __name__ == '__main__':
    app.run_server(debug=True)

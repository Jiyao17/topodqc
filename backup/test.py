

def swap(f1, c1, f2, c2, p=0.75):

    f = f1*f2 + (1-f1)*(1-f2)
    c = (c1 + c2) / p

    return f, c

def purify(f1, c1, f2, c2):
    p = f1*f2 + (1-f1)*(1-f2)

    f = f1*f2 / p
    c = (c1 + c2) / p

    return f, c


if __name__ == "__main__":

    f1, c1 = 0.85, 1
    f2, c2 = 0.82, 1
    f3, c3 = 0.84, 1
    f4, c4 = 0.88, 1

    # scheme 1
    f12, c12 = swap(f1, c1, f2, c2)
    print("f12, c12:", f12, c12)
    f3, c3 = purify(f3, c3, f3, c3)
    print("f3 purified:", f3, c3)
    f34, c34 = swap(f3, c3, f4, c4)
    print("f34, c34:", f34, c34)
    f, c = swap(f12, c12, f34, c34)
    print("Final result:", f, c)

    f1, c1 = 0.85, 1
    f2, c2 = 0.82, 1
    f3, c3 = 0.84, 1
    f4, c4 = 0.88, 1
    # scheme 2
    f12, c12 = swap(f1, c1, f2, c2)
    print("f12, c12:", f12, c12)
    f34, c34 = swap(f3, c3, f4, c4)
    print("f34, c34:", f34, c34)
    # f34, c34 = purify(f34, c34, f34, c34)
    # print("f34 purified:", f34, c34)
    f, c = swap(f12, c12, f34, c34)
    print("Final result:", f, c)


    f1, c1 = 0.85, 1
    f2, c2 = 0.82, 1
    f3, c3 = 0.84, 1
    f4, c4 = 0.88, 1
    # scheme 3
    f12, c12 = swap(f1, c1, f2, c2)
    print("f12, c12:", f12, c12)
    f3, c3 = purify(f3, c3, f3, c3)
    print("f3 purified:", f3, c3)
    f4, c4 = purify(f4, c4, f4, c4)
    print("f4 purified:", f4, c4)
    f34, c34 = swap(f3, c3, f4, c4)
    print("f34, c34:", f34, c34)
    f, c = swap(f12, c12, f34, c34)
    print("Final result:", f, c)
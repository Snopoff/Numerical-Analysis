filename = 'data.txt'


def write_data():
    """Function devoted to filling input data file with X points"""
    with open(filename, 'r') as f:
        lastline = f.readlines()[-1]

    a, b, n = list(map(lambda x: float(x), lastline.split(" ")))
    h = (b-a) / n
    with open(filename, 'a') as f:
        for i in range(0, int(n)):
            f.write("\n" + " ".join([str(i), str(a + i*h)]))


def main():
    """Main function"""
    write_data()


if __name__ == '__main__':
    main()

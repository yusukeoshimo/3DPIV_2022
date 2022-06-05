def join(*args):
    join_path = ''
    for i, arg in enumerate(args):
        if i == 0:
            join_path += arg
        else:
            join_path += '\\'+arg
    return join_path

if __name__ == '__main__':
    print(join('c:', 'aaa', 'bbb'))
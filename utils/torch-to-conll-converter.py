from sys import argv
from os import system


def run(line_pos_file, predicts_file, write_predicts_file, original_test_file):
    orig_to_new = {}
    with open(line_pos_file, 'r') as f:
        for line in f:
            orig, new = line.strip().split('\t')
            orig_to_new[int(orig) - 1] = int(new) - 1

    new_preds = {}
    with open(predicts_file, 'r') as f:
        for ind, line in enumerate(f):
            new_preds[ind] = line.strip()

    out = open(write_predicts_file, 'w')

    with open(original_test_file, 'r') as f:
        for ind, line in enumerate(f):
            stripped = line.strip()
            new_line = ''
            if stripped.split(' ')[0] == '-DOCSTART-':
                new_line = stripped + ' O'
            elif stripped != '':
                new = orig_to_new[ind]
                pred = new_preds[new]
                new_line = stripped + ' ' + pred
            out.write(new_line + '\n')
    out.close()

def convert_from_iobes(filename):
    system('cp {0} {1}-2'.format(filename, filename))
    cmd = 'cat {0} | python utils/iob-iobes.py true iobes | python utils/iob-iobes.py true iob > {1}-1'.format(filename, filename)
    system(cmd)
    system('cp {0}-1 {1}'.format(filename, filename))

if __name__ == '__main__':
    predicts_file = argv[1]
    original_test_file = argv[2]
    line_pos_file = '{0}.index-line-pos'.format(original_test_file)
    write_predicts_file = '{0}-predicted'.format(original_test_file)
    run(line_pos_file, predicts_file, write_predicts_file, original_test_file)
    convert_from_iobes(write_predicts_file)
    print(write_predicts_file)

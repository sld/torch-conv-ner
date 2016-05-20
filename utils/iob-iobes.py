from sys import argv, stdin, stdout

# BIO -> BIOES

# B O -> S O
# B B -> S B|S
# I O -> E O


def tag_name(full):
  if full == 'O':
    return full
  else:
    return full[2:]

def from_iobes():
  def get_new(true, prev_true, equal_types):
    if true == 'S' and equal_types:
      return 'B'
    elif true == 'S' and not equal_types:
      return 'B'
    elif true == 'B' and not equal_types:
      return 'B'
    elif true == 'B' and equal_types:
      return 'B'
    elif true == 'E':
      return 'I'
    else:
      return true

  lines = stdin.readlines()
  for ind, line in enumerate(lines):
    if line != '\n':
      splitted = line.strip().split(' ')
      true = splitted[-2][0]
      pred = splitted[-1][0]

      if ind != 0:
        if lines[ind-1] != '\n':
          prev_true = lines[ind-1].strip().split(' ')[-2]
          prev_pred = lines[ind-1].strip().split(' ')[-1]
        else:
          prev_true = ['']
          prev_pred = ['']
        new_true = get_new(true[0], prev_true[0], tag_name(prev_true) == tag_name(true))
        new_pred = get_new(pred[0], prev_pred[0], tag_name(prev_pred) == tag_name(pred))
        splitted[-2] = new_true + splitted[-2][1:]
        splitted[-1] = new_pred + splitted[-1][1:]

      joined = ' '.join(splitted)
    else:
      joined = ''
    print(joined)

def to_iobes():
  current = ''
  next_ = ''
  lines = stdin.readlines()
  for ind, line in enumerate(lines):
    if line != '\n':
      splitted = line.strip().split(' ')

      current = splitted[-1][0]
      new_current = ''
      next_ = lines[ind+1].strip().split(' ')[-1]
      if len(next_) > 0:
        next_ = next_[0]

      if current == 'B' and next_ == 'O':
        new_current = 'S'
      elif current == 'B' and next_ == 'B':
        new_current = 'S'
      elif current == 'I' and next_ == 'O':
        new_current = 'E'
      elif current == 'I' and next_ == 'B':
        new_current = 'E'
      elif current == 'B' and next_ == '':
        new_current = 'S'
      elif current == 'I' and next_ == '':
        new_current = 'E'
      else:
        new_current = current[0]
      splitted[-1] = new_current + splitted[-1][1:]

      joined = ' '.join(splitted)
    else:
      joined = ''
    print(joined)


def from_iob():
  previous = None
  for line in stdin:
    if line == '\n':
      if previous is not None:
        print(' '.join(previous))
      print('')
      previous = None
      continue
    words = line.strip().split()

    pred_tag =  words[-1]
    true_tag =  words[-2]

    if true_tag[0] == 'B' and (previous == None or previous[-2] == 'O'):
      words[-2] = 'I' + true_tag[1:]
    elif true_tag[0] == 'B' and (previous is not None and tag_name(true_tag) == tag_name(previous[-2])):
      words[-2] = 'B' + true_tag[1:]

    if pred_tag[0] == 'B' and (previous == None or previous[-1] == 'O'):
      words[-1] = 'I' + pred_tag[1:]
    elif pred_tag[0] == 'B' and (previous is not None and tag_name(pred_tag) == tag_name(previous[-1])):
      words[-1] = 'B' + pred_tag[1:]

    if previous:
      print(' '.join(previous))
    previous = words
  if previous:
    print(' '.join(previous))

def to_iob():
  previous = None
  for line in stdin:
    if line == '\n':
      if previous is not None:
        print(' '.join(previous))
      print('')
      previous = None
      continue
    words = line.split()
    word = words[0]
    tag =  words[-1]
    if tag[0] == 'I' and (previous == None or previous[-1] == 'O'):
      words[-1] = 'B' + tag[1:]
    elif tag[0] == 'I' and tag_name(tag) != tag_name(previous[-1]):
      words[-1] = 'B' + tag[1:]
    if previous:
      print(' '.join(previous))
    previous = words
  if previous:
    print(' '.join(previous))

if __name__ == '__main__':
  reverse = (True if argv[1] == 'true' else False)
  format_ = argv[2]
  if format_ == 'iob':
    (from_iob() if reverse else to_iob())
  elif format_ == 'iobes':
    (from_iobes() if reverse else to_iobes())

import logging
import os
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_integer('x', 16, '')
flags.DEFINE_integer('y', 4, '')
flags.register_validator('y', lambda z: FLAGS.x % z == 0, message='--y does not divide x!')
QR = 450
flags.DEFINE_integer('z', QR, 'see if flags runs after some code')
flags.DEFINE_multi_integer('cutoffs', [], 'variable list')
#flags.register_validator('cutoffs', lambda x: (len(x)<5 and all([i<200 for i in x])), message= 'Can\'t have more than 5 and none can be over 2000')

def main(argv):
    print("x =", FLAGS.x)
    print("y =", FLAGS.y)
    print('z =', FLAGS.z)
#    print('Cutoffs =', FLAGS.ctutoffs)
    logging.warning('WARNING! There\'s nothing wrong')
    logging.info('This should only print with --log')
    logging.info(f'Will an f string work? {4+5}')
    os.system('ls')

if __name__=="__main__":
    app.run(main)

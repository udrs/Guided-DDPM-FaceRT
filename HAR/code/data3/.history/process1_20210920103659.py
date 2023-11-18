# from numpy.core.shape_base import vstack
from numpy.lib.function_base import sinc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def load_data(file_path):
    data = pd.read_csv(file_path, sep="\s+")
    return data

train_paths = [['Test1\\20210627_091241\\AccX_0.txt',
                'Test1\\20210627_091241\\AccY_0.txt',
                'Test1\\20210627_091241\\AccZ_0.txt',
                'Test1\\20210627_091241\\GyroX_0.txt',
                'Test1\\20210627_091241\\GyroY_0.txt',
                'Test1\\20210627_091241\\GyroZ_0.txt',
                'Test1\\20210627_091241\\Label_0.txt'],
                
                ['Test1\\20210627_091523\\AccX_0.txt',
                'Test1\\20210627_091523\\AccY_0.txt',
                'Test1\\20210627_091523\\AccZ_0.txt',
                'Test1\\20210627_091523\\GyroX_0.txt',
                'Test1\\20210627_091523\\GyroY_0.txt',
                'Test1\\20210627_091523\\GyroZ_0.txt',
                'Test1\\20210627_091523\\Label_0.txt'],

                # ['Test1\\20210627_092141\\AccX_0.txt',
                # 'Test1\\20210627_092141\\AccY_0.txt',
                # 'Test1\\20210627_092141\\AccZ_0.txt',
                # 'Test1\\20210627_092141\\GyroX_0.txt',
                # 'Test1\\20210627_092141\\GyroY_0.txt',
                # 'Test1\\20210627_092141\\GyroZ_0.txt',
                # 'Test1\\20210627_092141\\Label_0.txt'],
                
                ['Test1\\20210627_092516\\AccX_0.txt',
                'Test1\\20210627_092516\\AccY_0.txt',
                'Test1\\20210627_092516\\AccZ_0.txt',
                'Test1\\20210627_092516\\GyroX_0.txt',
                'Test1\\20210627_092516\\GyroY_0.txt',
                'Test1\\20210627_092516\\GyroZ_0.txt',
                'Test1\\20210627_092516\\Label_0.txt'],

                ['Test1\\20210627_092828\\AccX_0.txt',
                'Test1\\20210627_092828\\AccY_0.txt',
                'Test1\\20210627_092828\\AccZ_0.txt',
                'Test1\\20210627_092828\\GyroX_0.txt',
                'Test1\\20210627_092828\\GyroY_0.txt',
                'Test1\\20210627_092828\\GyroZ_0.txt',
                'Test1\\20210627_092828\\Label_0.txt'],

                ['Test1\\20210627_093216\\AccX_0.txt',
                'Test1\\20210627_093216\\AccY_0.txt',
                'Test1\\20210627_093216\\AccZ_0.txt',
                'Test1\\20210627_093216\\GyroX_0.txt',
                'Test1\\20210627_093216\\GyroY_0.txt',
                'Test1\\20210627_093216\\GyroZ_0.txt',
                'Test1\\20210627_093216\\Label_0.txt'],

                ['Test1\\20210627_094333\\AccX_0.txt',
                'Test1\\20210627_094333\\AccY_0.txt',
                'Test1\\20210627_094333\\AccZ_0.txt',
                'Test1\\20210627_094333\\GyroX_0.txt',
                'Test1\\20210627_094333\\GyroY_0.txt',
                'Test1\\20210627_094333\\GyroZ_0.txt',
                'Test1\\20210627_094333\\Label_0.txt'],

                ['Test1\\20210627_094652\\AccX_0.txt',
                'Test1\\20210627_094652\\AccY_0.txt',
                'Test1\\20210627_094652\\AccZ_0.txt',
                'Test1\\20210627_094652\\GyroX_0.txt',
                'Test1\\20210627_094652\\GyroY_0.txt',
                'Test1\\20210627_094652\\GyroZ_0.txt',
                'Test1\\20210627_094652\\Label_0.txt'],

                ['Test1\\20210627_094843\\AccX_0.txt',
                'Test1\\20210627_094843\\AccY_0.txt',
                'Test1\\20210627_094843\\AccZ_0.txt',
                'Test1\\20210627_094843\\GyroX_0.txt',
                'Test1\\20210627_094843\\GyroY_0.txt',
                'Test1\\20210627_094843\\GyroZ_0.txt',
                'Test1\\20210627_094843\\Label_0.txt'],

                ['Test1\\20210627_095046\\AccX_0.txt',
                'Test1\\20210627_095046\\AccY_0.txt',
                'Test1\\20210627_095046\\AccZ_0.txt',
                'Test1\\20210627_095046\\GyroX_0.txt',
                'Test1\\20210627_095046\\GyroY_0.txt',
                'Test1\\20210627_095046\\GyroZ_0.txt',
                'Test1\\20210627_095046\\Label_0.txt'],
                
                
                ['Test1\\20210627_095333\\AccX_0.txt',
                'Test1\\20210627_095333\\AccY_0.txt',
                'Test1\\20210627_095333\\AccZ_0.txt',
                'Test1\\20210627_095333\\GyroX_0.txt',
                'Test1\\20210627_095333\\GyroY_0.txt',
                'Test1\\20210627_095333\\GyroZ_0.txt',
                'Test1\\20210627_095333\\Label_0.txt'],

                ['Test1\\20210627_095640\\AccX_0.txt',
                'Test1\\20210627_095640\\AccY_0.txt',
                'Test1\\20210627_095640\\AccZ_0.txt',
                'Test1\\20210627_095640\\GyroX_0.txt',
                'Test1\\20210627_095640\\GyroY_0.txt',
                'Test1\\20210627_095640\\GyroZ_0.txt',
                'Test1\\20210627_095640\\Label_0.txt'],

                ['Test1\\20210627_095855\\AccX_0.txt',
                'Test1\\20210627_095855\\AccY_0.txt',
                'Test1\\20210627_095855\\AccZ_0.txt',
                'Test1\\20210627_095855\\GyroX_0.txt',
                'Test1\\20210627_095855\\GyroY_0.txt',
                'Test1\\20210627_095855\\GyroZ_0.txt',
                'Test1\\20210627_095855\\Label_0.txt'],

                ['Test1\\20210627_100045\\AccX_0.txt',
                'Test1\\20210627_100045\\AccY_0.txt',
                'Test1\\20210627_100045\\AccZ_0.txt',
                'Test1\\20210627_100045\\GyroX_0.txt',
                'Test1\\20210627_100045\\GyroY_0.txt',
                'Test1\\20210627_100045\\GyroZ_0.txt',
                'Test1\\20210627_100045\\Label_0.txt'],

                ['Test1\\20210627_100644\\AccX_0.txt',
                'Test1\\20210627_100644\\AccY_0.txt',
                'Test1\\20210627_100644\\AccZ_0.txt',
                'Test1\\20210627_100644\\GyroX_0.txt',
                'Test1\\20210627_100644\\GyroY_0.txt',
                'Test1\\20210627_100644\\GyroZ_0.txt',
                'Test1\\20210627_100644\\Label_0.txt'],

                ['Test1\\20210627_100830\\AccX_0.txt',
                'Test1\\20210627_100830\\AccY_0.txt',
                'Test1\\20210627_100830\\AccZ_0.txt',
                'Test1\\20210627_100830\\GyroX_0.txt',
                'Test1\\20210627_100830\\GyroY_0.txt',
                'Test1\\20210627_100830\\GyroZ_0.txt',
                'Test1\\20210627_100830\\Label_0.txt'],

                ['Test1\\20210627_101102\\AccX_0.txt',
                'Test1\\20210627_101102\\AccY_0.txt',
                'Test1\\20210627_101102\\AccZ_0.txt',
                'Test1\\20210627_101102\\GyroX_0.txt',
                'Test1\\20210627_101102\\GyroY_0.txt',
                'Test1\\20210627_101102\\GyroZ_0.txt',
                'Test1\\20210627_101102\\Label_0.txt'],

                ['Test1\\20210627_101244\\AccX_0.txt',
                'Test1\\20210627_101244\\AccY_0.txt',
                'Test1\\20210627_101244\\AccZ_0.txt',
                'Test1\\20210627_101244\\GyroX_0.txt',
                'Test1\\20210627_101244\\GyroY_0.txt',
                'Test1\\20210627_101244\\GyroZ_0.txt',
                'Test1\\20210627_101244\\Label_0.txt'],

                ['Test1\\20210627_101438\\AccX_0.txt',
                'Test1\\20210627_101438\\AccY_0.txt',
                'Test1\\20210627_101438\\AccZ_0.txt',
                'Test1\\20210627_101438\\GyroX_0.txt',
                'Test1\\20210627_101438\\GyroY_0.txt',
                'Test1\\20210627_101438\\GyroZ_0.txt',
                'Test1\\20210627_101438\\Label_0.txt'],

                ['Test1\\20210627_101616\AccX_0.txt',
                'Test1\\20210627_101616\AccY_0.txt',
                'Test1\\20210627_101616\AccZ_0.txt',
                'Test1\\20210627_101616\GyroX_0.txt',
                'Test1\\20210627_101616\GyroY_0.txt',
                'Test1\\20210627_101616\GyroZ_0.txt',
                'Test1\\20210627_101616\Label_0.txt'],

                ['Turn2\\20210627_112628\\AccX_0.txt',
                'Turn2\\20210627_112628\\AccY_0.txt',
                'Turn2\\20210627_112628\\AccZ_0.txt',
                'Turn2\\20210627_112628\\GyroX_0.txt',
                'Turn2\\20210627_112628\\GyroY_0.txt',
                'Turn2\\20210627_112628\\GyroZ_0.txt',
                'Turn2\\20210627_112628\\Label_0.txt'],

                ['Turn2\\20210627_112711\\AccX_0.txt',
                'Turn2\\20210627_112711\\AccY_0.txt',
                'Turn2\\20210627_112711\\AccZ_0.txt',
                'Turn2\\20210627_112711\\GyroX_0.txt',
                'Turn2\\20210627_112711\\GyroY_0.txt',
                'Turn2\\20210627_112711\\GyroZ_0.txt',
                'Turn2\\20210627_112711\\Label_0.txt'],

                ['Turn2\\20210627_112742\\AccX_0.txt',
                'Turn2\\20210627_112742\\AccY_0.txt',
                'Turn2\\20210627_112742\\AccZ_0.txt',
                'Turn2\\20210627_112742\\GyroX_0.txt',
                'Turn2\\20210627_112742\\GyroY_0.txt',
                'Turn2\\20210627_112742\\GyroZ_0.txt',
                'Turn2\\20210627_112742\\Label_0.txt'],

                ['Turn2\\20210627_112804\\AccX_0.txt',
                'Turn2\\20210627_112804\\AccY_0.txt',
                'Turn2\\20210627_112804\\AccZ_0.txt',
                'Turn2\\20210627_112804\\GyroX_0.txt',
                'Turn2\\20210627_112804\\GyroY_0.txt',
                'Turn2\\20210627_112804\\GyroZ_0.txt',
                'Turn2\\20210627_112804\\Label_0.txt'],

                ['Turn2\\20210627_112833\\AccX_0.txt',
                'Turn2\\20210627_112833\\AccY_0.txt',
                'Turn2\\20210627_112833\\AccZ_0.txt',
                'Turn2\\20210627_112833\\GyroX_0.txt',
                'Turn2\\20210627_112833\\GyroY_0.txt',
                'Turn2\\20210627_112833\\GyroZ_0.txt',
                'Turn2\\20210627_112833\\Label_0.txt'],

                ['Turn2\\20210627_112923\\AccX_0.txt',
                'Turn2\\20210627_112923\\AccY_0.txt',
                'Turn2\\20210627_112923\\AccZ_0.txt',
                'Turn2\\20210627_112923\\GyroX_0.txt',
                'Turn2\\20210627_112923\\GyroY_0.txt',
                'Turn2\\20210627_112923\\GyroZ_0.txt',
                'Turn2\\20210627_112923\\Label_0.txt'],

                ['Turn2\\20210627_112946\\AccX_0.txt',
                'Turn2\\20210627_112946\\AccY_0.txt',
                'Turn2\\20210627_112946\\AccZ_0.txt',
                'Turn2\\20210627_112946\\GyroX_0.txt',
                'Turn2\\20210627_112946\\GyroY_0.txt',
                'Turn2\\20210627_112946\\GyroZ_0.txt',
                'Turn2\\20210627_112946\\Label_0.txt'],

                ['Turn2\\20210627_113008\\AccX_0.txt',
                'Turn2\\20210627_113008\\AccY_0.txt',
                'Turn2\\20210627_113008\\AccZ_0.txt',
                'Turn2\\20210627_113008\\GyroX_0.txt',
                'Turn2\\20210627_113008\\GyroY_0.txt',
                'Turn2\\20210627_113008\\GyroZ_0.txt',
                'Turn2\\20210627_113008\\Label_0.txt'],
                
                ['Turn2\\20210627_113043\\AccX_0.txt',
                'Turn2\\20210627_113043\\AccY_0.txt',
                'Turn2\\20210627_113043\\AccZ_0.txt',
                'Turn2\\20210627_113043\\GyroX_0.txt',
                'Turn2\\20210627_113043\\GyroY_0.txt',
                'Turn2\\20210627_113043\\GyroZ_0.txt',
                'Turn2\\20210627_113043\\Label_0.txt'],

                ['Turn2\\20210627_113100\\AccX_0.txt',
                'Turn2\\20210627_113100\\AccY_0.txt',
                'Turn2\\20210627_113100\\AccZ_0.txt',
                'Turn2\\20210627_113100\\GyroX_0.txt',
                'Turn2\\20210627_113100\\GyroY_0.txt',
                'Turn2\\20210627_113100\\GyroZ_0.txt',
                'Turn2\\20210627_113100\\Label_0.txt'],

                ['TurnRight\\20210627_112037\\AccX_0.txt',
                'TurnRight\\20210627_112037\\AccY_0.txt',
                'TurnRight\\20210627_112037\\AccZ_0.txt',
                'TurnRight\\20210627_112037\\GyroX_0.txt',
                'TurnRight\\20210627_112037\\GyroY_0.txt',
                'TurnRight\\20210627_112037\\GyroZ_0.txt',
                'TurnRight\\20210627_112037\\Label_0.txt'],

                ['TurnRight\\20210627_112107\\AccX_0.txt',
                'TurnRight\\20210627_112107\\AccY_0.txt',
                'TurnRight\\20210627_112107\\AccZ_0.txt',
                'TurnRight\\20210627_112107\\GyroX_0.txt',
                'TurnRight\\20210627_112107\\GyroY_0.txt',
                'TurnRight\\20210627_112107\\GyroZ_0.txt',
                'TurnRight\\20210627_112107\\Label_0.txt'],

                ['TurnRight\\20210627_112137\\AccX_0.txt',
                'TurnRight\\20210627_112137\\AccY_0.txt',
                'TurnRight\\20210627_112137\\AccZ_0.txt',
                'TurnRight\\20210627_112137\\GyroX_0.txt',
                'TurnRight\\20210627_112137\\GyroY_0.txt',
                'TurnRight\\20210627_112137\\GyroZ_0.txt',
                'TurnRight\\20210627_112137\\Label_0.txt'],

                ['TurnRight\\20210627_112200\\AccX_0.txt',
                'TurnRight\\20210627_112200\\AccY_0.txt',
                'TurnRight\\20210627_112200\\AccZ_0.txt',
                'TurnRight\\20210627_112200\\GyroX_0.txt',
                'TurnRight\\20210627_112200\\GyroY_0.txt',
                'TurnRight\\20210627_112200\\GyroZ_0.txt',
                'TurnRight\\20210627_112200\\Label_0.txt'],

                ['TurnRight\\20210627_112245\\AccX_0.txt',
                'TurnRight\\20210627_112245\\AccY_0.txt',
                'TurnRight\\20210627_112245\\AccZ_0.txt',
                'TurnRight\\20210627_112245\\GyroX_0.txt',
                'TurnRight\\20210627_112245\\GyroY_0.txt',
                'TurnRight\\20210627_112245\\GyroZ_0.txt',
                'TurnRight\\20210627_112245\\Label_0.txt'],

                ['TurnRight\\20210627_112304\\AccX_0.txt',
                'TurnRight\\20210627_112304\\AccY_0.txt',
                'TurnRight\\20210627_112304\\AccZ_0.txt',
                'TurnRight\\20210627_112304\\GyroX_0.txt',
                'TurnRight\\20210627_112304\\GyroY_0.txt',
                'TurnRight\\20210627_112304\\GyroZ_0.txt',
                'TurnRight\\20210627_112304\\Label_0.txt'],

                ['TurnRight\\20210627_112329\\AccX_0.txt',
                'TurnRight\\20210627_112329\\AccY_0.txt',
                'TurnRight\\20210627_112329\\AccZ_0.txt',
                'TurnRight\\20210627_112329\\GyroX_0.txt',
                'TurnRight\\20210627_112329\\GyroY_0.txt',
                'TurnRight\\20210627_112329\\GyroZ_0.txt',
                'TurnRight\\20210627_112329\\Label_0.txt'],


                ['TurnRight\\20210627_112416\\AccX_0.txt',
                'TurnRight\\20210627_112416\\AccY_0.txt',
                'TurnRight\\20210627_112416\\AccZ_0.txt',
                'TurnRight\\20210627_112416\\GyroX_0.txt',
                'TurnRight\\20210627_112416\\GyroY_0.txt',
                'TurnRight\\20210627_112416\\GyroZ_0.txt',
                'TurnRight\\20210627_112416\\Label_0.txt'],

                ['TurnRight\\20210627_112438\\AccX_0.txt',
                'TurnRight\\20210627_112438\\AccY_0.txt',
                'TurnRight\\20210627_112438\\AccZ_0.txt',
                'TurnRight\\20210627_112438\\GyroX_0.txt',
                'TurnRight\\20210627_112438\\GyroY_0.txt',
                'TurnRight\\20210627_112438\\GyroZ_0.txt',
                'TurnRight\\20210627_112438\\Label_0.txt'],

                ['TurnRight\\20210627_112506\\AccX_0.txt',
                'TurnRight\\20210627_112506\\AccY_0.txt',
                'TurnRight\\20210627_112506\\AccZ_0.txt',
                'TurnRight\\20210627_112506\\GyroX_0.txt',
                'TurnRight\\20210627_112506\\GyroY_0.txt',
                'TurnRight\\20210627_112506\\GyroZ_0.txt',
                'TurnRight\\20210627_112506\\Label_0.txt'],

                ['UpTurnUp\\20210627_105459\\AccX_0.txt',
                'UpTurnUp\\20210627_105459\\AccY_0.txt',
                'UpTurnUp\\20210627_105459\\AccZ_0.txt',
                'UpTurnUp\\20210627_105459\\GyroX_0.txt',
                'UpTurnUp\\20210627_105459\\GyroY_0.txt',
                'UpTurnUp\\20210627_105459\\GyroZ_0.txt',
                'UpTurnUp\\20210627_105459\\Label_0.txt'],

                ['UpTurnUp\\20210627_105529\\AccX_0.txt',
                'UpTurnUp\\20210627_105529\\AccY_0.txt',
                'UpTurnUp\\20210627_105529\\AccZ_0.txt',
                'UpTurnUp\\20210627_105529\\GyroX_0.txt',
                'UpTurnUp\\20210627_105529\\GyroY_0.txt',
                'UpTurnUp\\20210627_105529\\GyroZ_0.txt',
                'UpTurnUp\\20210627_105529\\Label_0.txt'],

                ['UpTurnUp\\20210627_105605\\AccX_0.txt',
                'UpTurnUp\\20210627_105605\\AccY_0.txt',
                'UpTurnUp\\20210627_105605\\AccZ_0.txt',
                'UpTurnUp\\20210627_105605\\GyroX_0.txt',
                'UpTurnUp\\20210627_105605\\GyroY_0.txt',
                'UpTurnUp\\20210627_105605\\GyroZ_0.txt',
                'UpTurnUp\\20210627_105605\\Label_0.txt'],


                ['UpTurnUp\\20210627_105605\\AccX_1.txt',
                'UpTurnUp\\20210627_105605\\AccY_1.txt',
                'UpTurnUp\\20210627_105605\\AccZ_1.txt',
                'UpTurnUp\\20210627_105605\\GyroX_1.txt',
                'UpTurnUp\\20210627_105605\\GyroY_1.txt',
                'UpTurnUp\\20210627_105605\\GyroZ_1.txt',
                'UpTurnUp\\20210627_105605\\Label_1.txt'],

                ['UpTurnUp\\20210627_105818\\AccX_0.txt',
                'UpTurnUp\\20210627_105818\\AccY_0.txt',
                'UpTurnUp\\20210627_105818\\AccZ_0.txt',
                'UpTurnUp\\20210627_105818\\GyroX_0.txt',
                'UpTurnUp\\20210627_105818\\GyroY_0.txt',
                'UpTurnUp\\20210627_105818\\GyroZ_0.txt',
                'UpTurnUp\\20210627_105818\\Label_0.txt'],


                ['UpTurnUp\\20210627_105932\\AccX_0.txt',
                'UpTurnUp\\20210627_105932\\AccY_0.txt',
                'UpTurnUp\\20210627_105932\\AccZ_0.txt',
                'UpTurnUp\\20210627_105932\\GyroX_0.txt',
                'UpTurnUp\\20210627_105932\\GyroY_0.txt',
                'UpTurnUp\\20210627_105932\\GyroZ_0.txt',
                'UpTurnUp\\20210627_105932\\Label_0.txt'],

                ['UpTurnUp\\20210627_110122\\AccX_0.txt',
                'UpTurnUp\\20210627_110122\\AccY_0.txt',
                'UpTurnUp\\20210627_110122\\AccZ_0.txt',
                'UpTurnUp\\20210627_110122\\GyroX_0.txt',
                'UpTurnUp\\20210627_110122\\GyroY_0.txt',
                'UpTurnUp\\20210627_110122\\GyroZ_0.txt',
                'UpTurnUp\\20210627_110122\\Label_0.txt'],

                ['UpTurnUp\\20210627_110149\\AccX_0.txt',
                'UpTurnUp\\20210627_110149\\AccY_0.txt',
                'UpTurnUp\\20210627_110149\\AccZ_0.txt',
                'UpTurnUp\\20210627_110149\\GyroX_0.txt',
                'UpTurnUp\\20210627_110149\\GyroY_0.txt',
                'UpTurnUp\\20210627_110149\\GyroZ_0.txt',
                'UpTurnUp\\20210627_110149\\Label_0.txt'],

                ['UpTurnUp\\20210627_110222\\AccX_0.txt',
                'UpTurnUp\\20210627_110222\\AccY_0.txt',
                'UpTurnUp\\20210627_110222\\AccZ_0.txt',
                'UpTurnUp\\20210627_110222\\GyroX_0.txt',
                'UpTurnUp\\20210627_110222\\GyroY_0.txt',
                'UpTurnUp\\20210627_110222\\GyroZ_0.txt',
                'UpTurnUp\\20210627_110222\\Label_0.txt'],

                ['UpTurnUp\\20210627_110256\\AccX_0.txt',
                'UpTurnUp\\20210627_110256\\AccY_0.txt',
                'UpTurnUp\\20210627_110256\\AccZ_0.txt',
                'UpTurnUp\\20210627_110256\\GyroX_0.txt',
                'UpTurnUp\\20210627_110256\\GyroY_0.txt',
                'UpTurnUp\\20210627_110256\\GyroZ_0.txt',
                'UpTurnUp\\20210627_110256\\Label_0.txt'],

                ['UpWalk\\20210627_111357\\AccX_0.txt',
                'UpWalk\\20210627_111357\\AccY_0.txt',
                'UpWalk\\20210627_111357\\AccZ_0.txt',
                'UpWalk\\20210627_111357\\GyroX_0.txt',
                'UpWalk\\20210627_111357\\GyroY_0.txt',
                'UpWalk\\20210627_111357\\GyroZ_0.txt',
                'UpWalk\\20210627_111357\\Label_0.txt'],

                ['UpWalk\\20210627_111423\\AccX_0.txt',
                'UpWalk\\20210627_111423\\AccY_0.txt',
                'UpWalk\\20210627_111423\\AccZ_0.txt',
                'UpWalk\\20210627_111423\\GyroX_0.txt',
                'UpWalk\\20210627_111423\\GyroY_0.txt',
                'UpWalk\\20210627_111423\\GyroZ_0.txt',
                'UpWalk\\20210627_111423\\Label_0.txt'],

                ['UpWalk\\20210627_111447\\AccX_0.txt',
                'UpWalk\\20210627_111447\\AccY_0.txt',
                'UpWalk\\20210627_111447\\AccZ_0.txt',
                'UpWalk\\20210627_111447\\GyroX_0.txt',
                'UpWalk\\20210627_111447\\GyroY_0.txt',
                'UpWalk\\20210627_111447\\GyroZ_0.txt',
                'UpWalk\\20210627_111447\\Label_0.txt'],

                ['UpWalk\\20210627_111518\\AccX_0.txt',
                'UpWalk\\20210627_111518\\AccY_0.txt',
                'UpWalk\\20210627_111518\\AccZ_0.txt',
                'UpWalk\\20210627_111518\\GyroX_0.txt',
                'UpWalk\\20210627_111518\\GyroY_0.txt',
                'UpWalk\\20210627_111518\\GyroZ_0.txt',
                'UpWalk\\20210627_111518\\Label_0.txt'],

                ['UpWalk\\20210627_111619\\AccX_0.txt',
                'UpWalk\\20210627_111619\\AccY_0.txt',
                'UpWalk\\20210627_111619\\AccZ_0.txt',
                'UpWalk\\20210627_111619\\GyroX_0.txt',
                'UpWalk\\20210627_111619\\GyroY_0.txt',
                'UpWalk\\20210627_111619\\GyroZ_0.txt',
                'UpWalk\\20210627_111619\\Label_0.txt'],

                ['UpWalk\\20210627_111642\\AccX_0.txt',
                'UpWalk\\20210627_111642\\AccY_0.txt',
                'UpWalk\\20210627_111642\\AccZ_0.txt',
                'UpWalk\\20210627_111642\\GyroX_0.txt',
                'UpWalk\\20210627_111642\\GyroY_0.txt',
                'UpWalk\\20210627_111642\\GyroZ_0.txt',
                'UpWalk\\20210627_111642\\Label_0.txt'],

                ['UpWalk\\20210627_111708\\AccX_0.txt',
                'UpWalk\\20210627_111708\\AccY_0.txt',
                'UpWalk\\20210627_111708\\AccZ_0.txt',
                'UpWalk\\20210627_111708\\GyroX_0.txt',
                'UpWalk\\20210627_111708\\GyroY_0.txt',
                'UpWalk\\20210627_111708\\GyroZ_0.txt',
                'UpWalk\\20210627_111708\\Label_0.txt'],

                ['UpWalk\\20210627_111741\\AccX_0.txt',
                'UpWalk\\20210627_111741\\AccY_0.txt',
                'UpWalk\\20210627_111741\\AccZ_0.txt',
                'UpWalk\\20210627_111741\\GyroX_0.txt',
                'UpWalk\\20210627_111741\\GyroY_0.txt',
                'UpWalk\\20210627_111741\\GyroZ_0.txt',
                'UpWalk\\20210627_111741\\Label_0.txt'],

                ['UpWalk\\20210627_111805\\AccX_0.txt',
                'UpWalk\\20210627_111805\\AccY_0.txt',
                'UpWalk\\20210627_111805\\AccZ_0.txt',
                'UpWalk\\20210627_111805\\GyroX_0.txt',
                'UpWalk\\20210627_111805\\GyroY_0.txt',
                'UpWalk\\20210627_111805\\GyroZ_0.txt',
                'UpWalk\\20210627_111805\\Label_0.txt'],

                ['UpWalk\\20210627_111840\\AccX_0.txt',
                'UpWalk\\20210627_111840\\AccY_0.txt',
                'UpWalk\\20210627_111840\\AccZ_0.txt',
                'UpWalk\\20210627_111840\\GyroX_0.txt',
                'UpWalk\\20210627_111840\\GyroY_0.txt',
                'UpWalk\\20210627_111840\\GyroZ_0.txt',
                'UpWalk\\20210627_111840\\Label_0.txt'],

                ['WalkUp\\20210627_110710\\AccX_0.txt',
                'WalkUp\\20210627_110710\\AccY_0.txt',
                'WalkUp\\20210627_110710\\AccZ_0.txt',
                'WalkUp\\20210627_110710\\GyroX_0.txt',
                'WalkUp\\20210627_110710\\GyroY_0.txt',
                'WalkUp\\20210627_110710\\GyroZ_0.txt',
                'WalkUp\\20210627_110710\\Label_0.txt'],

                ['WalkUp\\20210627_110814\\AccX_0.txt',
                'WalkUp\\20210627_110814\\AccY_0.txt',
                'WalkUp\\20210627_110814\\AccZ_0.txt',
                'WalkUp\\20210627_110814\\GyroX_0.txt',
                'WalkUp\\20210627_110814\\GyroY_0.txt',
                'WalkUp\\20210627_110814\\GyroZ_0.txt',
                'WalkUp\\20210627_110814\\Label_0.txt'],

                ['WalkUp\\20210627_110844\\AccX_0.txt',
                'WalkUp\\20210627_110844\\AccY_0.txt',
                'WalkUp\\20210627_110844\\AccZ_0.txt',
                'WalkUp\\20210627_110844\\GyroX_0.txt',
                'WalkUp\\20210627_110844\\GyroY_0.txt',
                'WalkUp\\20210627_110844\\GyroZ_0.txt',
                'WalkUp\\20210627_110844\\Label_0.txt'],

                ['WalkUp\\20210627_110915\\AccX_0.txt',
                'WalkUp\\20210627_110915\\AccY_0.txt',
                'WalkUp\\20210627_110915\\AccZ_0.txt',
                'WalkUp\\20210627_110915\\GyroX_0.txt',
                'WalkUp\\20210627_110915\\GyroY_0.txt',
                'WalkUp\\20210627_110915\\GyroZ_0.txt',
                'WalkUp\\20210627_110915\\Label_0.txt'],

                ['WalkUp\\20210627_110946\\AccX_0.txt',
                'WalkUp\\20210627_110946\\AccY_0.txt',
                'WalkUp\\20210627_110946\\AccZ_0.txt',
                'WalkUp\\20210627_110946\\GyroX_0.txt',
                'WalkUp\\20210627_110946\\GyroY_0.txt',
                'WalkUp\\20210627_110946\\GyroZ_0.txt',
                'WalkUp\\20210627_110946\\Label_0.txt'],

                ['WalkUp\\20210627_111013\\AccX_0.txt',
                'WalkUp\\20210627_111013\\AccY_0.txt',
                'WalkUp\\20210627_111013\\AccZ_0.txt',
                'WalkUp\\20210627_111013\\GyroX_0.txt',
                'WalkUp\\20210627_111013\\GyroY_0.txt',
                'WalkUp\\20210627_111013\\GyroZ_0.txt',
                'WalkUp\\20210627_111013\\Label_0.txt'],

                # ['WalkUp\\20210627_111041\AccX_0.txt',
                # 'WalkUp\\20210627_111041\AccY_0.txt',
                # 'WalkUp\\20210627_111041\AccZ_0.txt',
                # 'WalkUp\\20210627_111041\GyroX_0.txt',
                # 'WalkUp\\20210627_111041\GyroY_0.txt',
                # 'WalkUp\\20210627_111041\GyroZ_0.txt',
                # 'WalkUp\\20210627_111041\Label_0.txt'],

                ['WalkUp\\20210627_111109\\AccX_0.txt',
                'WalkUp\\20210627_111109\\AccY_0.txt',
                'WalkUp\\20210627_111109\\AccZ_0.txt',
                'WalkUp\\20210627_111109\\GyroX_0.txt',
                'WalkUp\\20210627_111109\\GyroY_0.txt',
                'WalkUp\\20210627_111109\\GyroZ_0.txt',
                'WalkUp\\20210627_111109\\Label_0.txt'],

                ['WalkUp\\20210627_111140\\AccX_0.txt',
                'WalkUp\\20210627_111140\\AccY_0.txt',
                'WalkUp\\20210627_111140\\AccZ_0.txt',
                'WalkUp\\20210627_111140\\GyroX_0.txt',
                'WalkUp\\20210627_111140\\GyroY_0.txt',
                'WalkUp\\20210627_111140\\GyroZ_0.txt',
                'WalkUp\\20210627_111140\\Label_0.txt'],

                ['WalkUp\\20210627_111213\\AccX_0.txt',
                'WalkUp\\20210627_111213\\AccY_0.txt',
                'WalkUp\\20210627_111213\\AccZ_0.txt',
                'WalkUp\\20210627_111213\\GyroX_0.txt',
                'WalkUp\\20210627_111213\\GyroY_0.txt',
                'WalkUp\\20210627_111213\\GyroZ_0.txt',
                'WalkUp\\20210627_111213\\Label_0.txt'],

                ['Exp\\20210627_101924\\AccX_0.txt',
                'Exp\\20210627_101924\\AccY_0.txt',
                'Exp\\20210627_101924\\AccZ_0.txt',
                'Exp\\20210627_101924\\GyroX_0.txt',
                'Exp\\20210627_101924\\GyroY_0.txt',
                'Exp\\20210627_101924\\GyroZ_0.txt',
                'Exp\\20210627_101924\\Label_0.txt'],

                ['Exp\\20210627_102255\\AccX_0.txt',
                'Exp\\20210627_102255\\AccY_0.txt',
                'Exp\\20210627_102255\\AccZ_0.txt',
                'Exp\\20210627_102255\\GyroX_0.txt',
                'Exp\\20210627_102255\\GyroY_0.txt',
                'Exp\\20210627_102255\\GyroZ_0.txt',
                'Exp\\20210627_102255\\Label_0.txt'],

                ['Exp\\20210627_102451\\AccX_0.txt',
                'Exp\\20210627_102451\\AccY_0.txt',
                'Exp\\20210627_102451\\AccZ_0.txt',
                'Exp\\20210627_102451\\GyroX_0.txt',
                'Exp\\20210627_102451\\GyroY_0.txt',
                'Exp\\20210627_102451\\GyroZ_0.txt',
                'Exp\\20210627_102451\\Label_0.txt'],

                ['Exp\\20210627_102901\\AccX_0.txt',
                'Exp\\20210627_102901\\AccY_0.txt',
                'Exp\\20210627_102901\\AccZ_0.txt',
                'Exp\\20210627_102901\\GyroX_0.txt',
                'Exp\\20210627_102901\\GyroY_0.txt',
                'Exp\\20210627_102901\\GyroZ_0.txt',
                'Exp\\20210627_102901\\Label_0.txt'],

                ['Exp\\20210627_103145\\AccX_0.txt',
                'Exp\\20210627_103145\\AccY_0.txt',
                'Exp\\20210627_103145\\AccZ_0.txt',
                'Exp\\20210627_103145\\GyroX_0.txt',
                'Exp\\20210627_103145\\GyroY_0.txt',
                'Exp\\20210627_103145\\GyroZ_0.txt',
                'Exp\\20210627_103145\\Label_0.txt'],

                ['Exp\\20210627_103328\\AccX_0.txt',
                'Exp\\20210627_103328\\AccY_0.txt',
                'Exp\\20210627_103328\\AccZ_0.txt',
                'Exp\\20210627_103328\\GyroX_0.txt',
                'Exp\\20210627_103328\\GyroY_0.txt',
                'Exp\\20210627_103328\\GyroZ_0.txt',
                'Exp\\20210627_103328\\Label_0.txt'],

                ['Exp\\20210627_103504\\AccX_0.txt',
                'Exp\\20210627_103504\\AccY_0.txt',
                'Exp\\20210627_103504\\AccZ_0.txt',
                'Exp\\20210627_103504\\GyroX_0.txt',
                'Exp\\20210627_103504\\GyroY_0.txt',
                'Exp\\20210627_103504\\GyroZ_0.txt',
                'Exp\\20210627_103504\\Label_0.txt'],

                ['Exp\\20210627_103643\\AccX_0.txt',
                'Exp\\20210627_103643\\AccY_0.txt',
                'Exp\\20210627_103643\\AccZ_0.txt',
                'Exp\\20210627_103643\\GyroX_0.txt',
                'Exp\\20210627_103643\\GyroY_0.txt',
                'Exp\\20210627_103643\\GyroZ_0.txt',
                'Exp\\20210627_103643\\Label_0.txt'],

                ['Exp\\20210627_103824\\AccX_0.txt',
                'Exp\\20210627_103824\\AccY_0.txt',
                'Exp\\20210627_103824\\AccZ_0.txt',
                'Exp\\20210627_103824\\GyroX_0.txt',
                'Exp\\20210627_103824\\GyroY_0.txt',
                'Exp\\20210627_103824\\GyroZ_0.txt',
                'Exp\\20210627_103824\\Label_0.txt'],

                ['Exp\\20210627_103959\\AccX_0.txt',
                'Exp\\20210627_103959\\AccY_0.txt',
                'Exp\\20210627_103959\\AccZ_0.txt',
                'Exp\\20210627_103959\\GyroX_0.txt',
                'Exp\\20210627_103959\\GyroY_0.txt',
                'Exp\\20210627_103959\\GyroZ_0.txt',
                'Exp\\20210627_103959\\Label_0.txt'], 

                ['Exp\\20210627_104220\\AccX_0.txt',
                'Exp\\20210627_104220\\AccY_0.txt',
                'Exp\\20210627_104220\\AccZ_0.txt',
                'Exp\\20210627_104220\\GyroX_0.txt',
                'Exp\\20210627_104220\\GyroY_0.txt',
                'Exp\\20210627_104220\\GyroZ_0.txt',
                'Exp\\20210627_104220\\Label_0.txt'],
                
                 ['Exp\\20210627_104407\\AccX_0.txt',
                'Exp\\20210627_104407\\AccY_0.txt',
                'Exp\\20210627_104407\\AccZ_0.txt',
                'Exp\\20210627_104407\\GyroX_0.txt',
                'Exp\\20210627_104407\\GyroY_0.txt',
                'Exp\\20210627_104407\\GyroZ_0.txt',
                'Exp\\20210627_104407\\Label_0.txt'],
                
                 ['Exp\\20210627_104550\\AccX_0.txt',
                'Exp\\20210627_104550\\AccY_0.txt',
                'Exp\\20210627_104550\\AccZ_0.txt',
                'Exp\\20210627_104550\\GyroX_0.txt',
                'Exp\\20210627_104550\\GyroY_0.txt',
                'Exp\\20210627_104550\\GyroZ_0.txt',
                'Exp\\20210627_104550\\Label_0.txt'],
                
                 ['Exp\\20210627_104736\\AccX_0.txt',
                'Exp\\20210627_104736\\AccY_0.txt',
                'Exp\\20210627_104736\\AccZ_0.txt',
                'Exp\\20210627_104736\\GyroX_0.txt',
                'Exp\\20210627_104736\\GyroY_0.txt',
                'Exp\\20210627_104736\\GyroZ_0.txt',
                'Exp\\20210627_104736\\Label_0.txt'] 
                          
                ]

Exp_paths = [ ['Exp2\\20210627_113254\\AccX_0.txt',
                'Exp2\\20210627_113254\\AccY_0.txt',
                'Exp2\\20210627_113254\\AccZ_0.txt',
                'Exp2\\20210627_113254\\GyroX_0.txt',
                'Exp2\\20210627_113254\\GyroY_0.txt',
                'Exp2\\20210627_113254\\GyroZ_0.txt',
                'Exp2\\20210627_113254\\Label_0.txt'],
                
                 ['Exp2\\20210627_113439\\AccX_0.txt',
                'Exp2\\20210627_113439\\AccY_0.txt',
                'Exp2\\20210627_113439\\AccZ_0.txt',
                'Exp2\\20210627_113439\\GyroX_0.txt',
                'Exp2\\20210627_113439\\GyroY_0.txt',
                'Exp2\\20210627_113439\\GyroZ_0.txt',
                'Exp2\\20210627_113439\\Label_0.txt'],
                
                 ['Exp2\\20210627_113820\\AccX_0.txt',
                'Exp2\\20210627_113820\\AccY_0.txt',
                'Exp2\\20210627_113820\\AccZ_0.txt',
                'Exp2\\20210627_113820\\GyroX_0.txt',
                'Exp2\\20210627_113820\\GyroY_0.txt',
                'Exp2\\20210627_113820\\GyroZ_0.txt',
                'Exp2\\20210627_113820\\Label_0.txt'],
                
                 ['Exp2\\20210627_113951\\AccX_0.txt',
                'Exp2\\20210627_113951\\AccY_0.txt',
                'Exp2\\20210627_113951\\AccZ_0.txt',
                'Exp2\\20210627_113951\\GyroX_0.txt',
                'Exp2\\20210627_113951\\GyroY_0.txt',
                'Exp2\\20210627_113951\\GyroZ_0.txt',
                'Exp2\\20210627_113951\\Label_0.txt'],
                
                 ['Exp\\20210627_104220\\AccX_0.txt',
                'Exp\\20210627_104220\\AccY_0.txt',
                'Exp\\20210627_104220\\AccZ_0.txt',
                'Exp\\20210627_104220\\GyroX_0.txt',
                'Exp\\20210627_104220\\GyroY_0.txt',
                'Exp\\20210627_104220\\GyroZ_0.txt',
                'Exp\\20210627_104220\\Label_0.txt'],
                
                 ['Exp\\20210627_104407\\AccX_0.txt',
                'Exp\\20210627_104407\\AccY_0.txt',
                'Exp\\20210627_104407\\AccZ_0.txt',
                'Exp\\20210627_104407\\GyroX_0.txt',
                'Exp\\20210627_104407\\GyroY_0.txt',
                'Exp\\20210627_104407\\GyroZ_0.txt',
                'Exp\\20210627_104407\\Label_0.txt'],
                
                 ['Exp\\20210627_104550\\AccX_0.txt',
                'Exp\\20210627_104550\\AccY_0.txt',
                'Exp\\20210627_104550\\AccZ_0.txt',
                'Exp\\20210627_104550\\GyroX_0.txt',
                'Exp\\20210627_104550\\GyroY_0.txt',
                'Exp\\20210627_104550\\GyroZ_0.txt',
                'Exp\\20210627_104550\\Label_0.txt'],
                
                 ['Exp\\20210627_104736\\AccX_0.txt',
                'Exp\\20210627_104736\\AccY_0.txt',
                'Exp\\20210627_104736\\AccZ_0.txt',
                'Exp\\20210627_104736\\GyroX_0.txt',
                'Exp\\20210627_104736\\GyroY_0.txt',
                'Exp\\20210627_104736\\GyroZ_0.txt',
                'Exp\\20210627_104736\\Label_0.txt']    
                ]

target_path_train = 'Train\\'

target_path_exp = 'Test_Exp\\'


# 所有原始的6-channel数据，变成 length*9
def process1(input_paths):
    for j in range(0,7):
        # Data = pd.read_csv(input_paths[i], header=None, index_col=None)
        # data = Data.values
        if (j<=2):
            if (j==0):
                Total_accX = pd.read_csv(input_paths[j], header=None, index_col=None)
                total_accX = Total_accX.values
                G = np.ones((total_accX.shape))
                G = G*9.81
                body_accX = total_accX - G
                length = total_accX.shape[0]
            elif(j==1):
                Total_accY = pd.read_csv(input_paths[j], header=None, index_col=None)
                total_accY = Total_accY.values
                G = np.ones((total_accY.shape))
                G = G*9.81
                body_accY = total_accY - G
            elif(j==2):
                Total_accZ = pd.read_csv(input_paths[j], header=None, index_col=None)
                total_accZ = Total_accZ.values
                G = np.ones((total_accZ.shape))
                G = G*9.81
                body_accZ = total_accZ - G
        elif(2<j<=5):
            if (j==3):
                Body_gyroX = pd.read_csv(input_paths[j], header=None, index_col=None)
                body_gyroX = Body_gyroX.values
                if (length > body_gyroX.shape[0]):

                    # print("gyro problem:")
                    # print('{} vs. {}'.format(length, body_accX.shape[0]))
                    length = body_gyroX.shape[0]
                    
                    # print(input_paths)
                    
            elif(j==4):
                Body_gyroY = pd.read_csv(input_paths[j], header=None, index_col=None)
                body_gyroY = Body_gyroY.values
            elif(j==5):
                Body_gyroZ = pd.read_csv(input_paths[j], header=None, index_col=None)
                body_gyroZ = Body_gyroZ.values
        elif(j==6):
            Label = pd.read_csv(input_paths[j], header=None, index_col=None)
            label = Label.values
            if (length > label.shape[0]):
                # print("label problem:")
                # print('{} vs. {}'.format(length, label.shape[0]))
                length = label.shape[0]
                # print(input_paths)

    pack = body_accX[:length,:].T
    pack = np.vstack((pack, body_accY[:length,:].T))
    pack = np.vstack((pack, body_accZ[:length,:].T))
    pack = np.vstack((pack, body_gyroX[:length,:].T))
    pack = np.vstack((pack, body_gyroY[:length,:].T))
    pack = np.vstack((pack, body_gyroZ[:length,:].T))
    pack = np.vstack((pack, total_accX[:length,:].T))
    pack = np.vstack((pack, total_accY[:length,:].T))
    pack = np.vstack((pack, total_accZ[:length,:].T))
    pack = np.vstack((pack, label[:length,:].T))
    return pack

def data_original(input_paths):
    inputs_length = len(input_paths)
    list = []

    for i in range(0, inputs_length):
        pack = process1(input_paths[i])
        list.append(pack)

    return list
 
############################################################ read and build orignal train_exp dataset ####################################
train1 = data_original(train_paths) ### list(numpy)
print('train size:', len(train1))

exp1 = data_original(Exp_paths)
print('exp length:', len(exp1))

# print('exp1 size:', exp1[0].shape)

# save original acc and gyro data in npz file
# np.savez("original_exp.npz", exp1)

############################################################# windowing cut #######################################################

def overlap(data_np, window_size=128, overlap=16):
    # print(data_np.shape)

    length = data_np.shape[1]
    start_point = 0
    stop_point = start_point + window_size
    overlap_data = data_np[:, start_point:stop_point]
    overlap_data = overlap_data[np.newaxis, :]
    # print('overlap size:', overlap_data.shape)

    while(stop_point <= length-overlap):
        start_point = start_point + overlap
        stop_point = start_point + window_size
        piece = data_np[:, start_point:stop_point]
        piece = piece[np.newaxis, :]
        overlap_data = np.concatenate((overlap_data, piece),axis=0)
        
    # print('overlap size:', overlap_data.shape)
    return overlap_data


for i in range(0, len(train1)):
    if (i==0):
        train2 = overlap(train1[i])
    else:
        pack = overlap(train1[i])
        train2 = np.concatenate((train2, pack))

print("train2 size:", train2.shape)

list_exp2 = []
for i in range(0, len(exp1)):
    if (i==0):
        exp2 = overlap(exp1[i],128,32)
    else:
        pack = overlap(exp1[i],128,32)
        exp2 = np.concatenate((exp2, pack))

    pack2 = overlap(exp1[i],128,32)
    list_exp2.append(pack2)

print("exp2 size:", exp2.shape)
print('exp size:', len(list_exp2))


############################################################### translate label #####################################

def getLabel2(input_label):
    PAIR = {1:[1,2], 
            2:[2,1],
            3:[1,3],
            4:[3,1],
            5:[1,4],
            6:[4,1],
            7:[1,5],
            8:[5,1],
            9:[3,5],
            10:[5,3],
            11:[4,5],
            12:[5,4],
            13:[1,1],
            14:[2,2],
            15:[3,3],
            16:[4,4],
            17:[5,5]}

    middel = []
    length1 = 0
    for label in input_label:
        if (len(middel)==0):
            middel.append(label)
        if (label != middel[0]):
            middel.append(label)
        if (len(middel) == 2):
            break
        length1 = length1 + 1

    if (len(middel)==1):
        middel.append(middel[0])
    
    # print('middle:', middel)

    label2 = list(PAIR.keys())[list(PAIR.values()).index(middel)]

    rate = length1/128

    rate = (100*rate)//10

    return label2, rate


def CreatLabel(data_np):

    label2_list = []
    rate_list = []
    for i in range(0, data_np.shape[0]):
        # print(data_np[i,9,:].shape)
        # print(data_np[i,9,:].flatten().shape)
        label_list = data_np[i,9,:].flatten().tolist()
    
        # for label in label_list:
        label2, rate = getLabel2(label_list)
        label2_list.append(label2)

        rate_list.append(rate)

    # print(label2_list)
    
    return label2_list, rate_list


train_label, rate_train_list = CreatLabel(train2)
print(len(train_label))

textfile = open("Train\\dataY.txt", "w+")
for element in train_label:
    textfile.write(str(element) + "\n")
textfile.close()

exp_label, rate_list_exp = CreatLabel(exp2)
textfile = open("Test_Exp\\dataY.txt", "w+")
for element in exp_label:
    textfile.write(str(element) + "\n")
textfile.close()

exp_label_list = []
for i in range(0, len(list_exp2)):
    exp2_label,rate_list_exp2 = CreatLabel(list_exp2[i])
    exp_label_list.append(exp2_label)


####################################################### build dataX file ######################################################

train_dataX = train2[:,0:9,:]

exp_dataX = exp2[:,0:9,:] 

exp_dataX_list = []
for i in range(0, len(list_exp2)):
    exp_dataX_list.append(list_exp2[i][:,0:9,:])

print(train_dataX.shape)
print(len(train_label))

print(exp_dataX.shape)
print(len(exp_label))

##################################### To filter five individual activity labels #######################

# train_label_individual = []
# exp_label_individual = []

# for i in range(0, len(train_label)):
#     if (train_label[i]==13):
#         label = 1
#     elif(train_label[i]==14):
#         label = 2
#     elif(train_label[i]==15):
#         label = 3
#     elif(train_label[i]==16):
#         label = 4
#     elif(train_label[i]==17):
#         label = 5
#     else:
#         label = 0 #### transition activity

#     if (i==0):
#         train_dataX_individual = train_dataX[0,:,:]
#         train_dataX_individual = np.reshape(train_dataX_individual,(1,9,128))
#         train_label_individual.append(label)
    
#     if(label!=0):
#         train_label_individual.append(label)
#         train_dataX_individual = np.concatenate((train_dataX_individual, np.reshape(train_dataX[i,:,:],(1,9,128))))

# print('individual train dataX shape:{}'.format(train_dataX_individual.shape))
# print(len(train_label_individual))

# for i in range(0, len(exp_label)):
#     if (exp_label[i]==13):
#         label = 1
#     elif(exp_label[i]==14):
#         label = 2
#     elif(exp_label[i]==15):
#         label = 3
#     elif(exp_label[i]==16):
#         label = 4
#     elif(exp_label[i]==17):
#         label = 5
#     else:
#         label = 0 #### transition activity

#     if (i==0):
#         exp_dataX_individual = exp_dataX[0,:,:]
#         exp_dataX_individual = np.reshape(exp_dataX_individual,(1,9,128))
#         exp_label_individual.append(label)
    
#     if(label!=0):
#         exp_label_individual.append(label)
#         exp_dataX_individual = np.concatenate((exp_dataX_individual, np.reshape(exp_dataX[i,:,:],(1,9,128))))

# print('individual exp dataX shape:{}'.format(exp_dataX_individual.shape))
# print(len(exp_label_individual))


####################################################### creat npz file of dataX and dataY ######################################

np.savez("Train\\train.npz", train_dataX, train_label)
np.savez("Test_Exp\\test.npz", exp_dataX, exp_label)

np.savez("Test_Exp\\exp0.npz", exp_dataX_list[0], exp_label_list[0])

print('exp0 size:', len(exp_label_list[0]))

np.savez("Test_Exp\\exp1.npz", exp_dataX_list[1], exp_label_list[])

print('exp0 size:', len(exp_label_list[1]))

np.savez("Test_Exp\\exp2.npz", exp_dataX_list[2], exp_label_list[2])

print('exp2 size:', len(exp_label_list[2]))

np.savez("Test_Exp\\exp3.npz", exp_dataX_list[3], exp_label_list[3])

print('exp3 size:', len(exp_label_list[3]))

np.savez("Test_Exp\\exp4.npz", exp_dataX_list[4], exp_label_list[4])

print('exp4 size:', len(exp_label_list[4]))


################ save rate file #####
textfile = open("Test_Exp\\test_rate.txt", "w")
for element in rate_list_exp:
    textfile.write(str(element) + "\n")
textfile.close()


############ Individual activity ##############
# np.savez("Train\\train_individual.npz", train_dataX_individual, train_label_individual)
# np.savez("Test_Exp\\test_individual.npz", exp_dataX_individual, exp_label_individual)


################################################### to get label3(W-U) signals and draw #######################

# rate_L3 = []

# for i in range(0, len(train_label)):
#     if (train_label[i]==3):
#         rate_L3.append(rate_list[i])
#         if (len(rate_L3)==1):
#             signals_L3 = np.reshape(train_dataX[i,:,:],(1,9,128))
#         else:
#             signals_L3 = np.concatenate((signals_L3, np.reshape(train_dataX[i,:,:],(1,9,128))))

# print(rate_L3)

# #### choose index 3: 0.515625 rate signals
# L3_body_accX = signals_L3[3,0,:].tolist()
# L3_body_accY = signals_L3[3,1,:].tolist()
# L3_body_accZ = signals_L3[3,2,:].tolist()
# L3_body_gyroX = signals_L3[3,3,:].tolist()
# L3_body_gyroY = signals_L3[3,4,:].tolist()
# L3_body_gyroZ = signals_L3[3,5,:].tolist()
# L3_total_accX = signals_L3[3,6,:].tolist()
# L3_total_accY = signals_L3[3,7,:].tolist()
# L3_total_accZ = signals_L3[3,8,:].tolist()

# x = np.arange(0,128)

# plt.plot(x, L3_body_accX)
# plt.plot(x, L3_body_accY)
# plt.plot(x, L3_body_accZ)
# plt.plot(x, L3_body_gyroX)
# plt.plot(x, L3_body_gyroY)
# plt.plot(x, L3_body_gyroZ)
# plt.plot(x, L3_total_accX)
# plt.plot(x, L3_total_accY)
# plt.plot(x, L3_total_accZ)
# plt.legend()
# plt.grid()
# plt.show()

############################################## to concatenate individual dataX ##########################
# PAIR = {1:[1,2], 
#     2:[2,1],
#     3:[1,3],
#     4:[3,1],
#     5:[1,4],
#     6:[4,1],
#     7:[1,5],
#     8:[5,1],
#     9:[3,5],
#     10:[4,5],
#     11:[1,1],
#     12:[2,2],
#     13:[3,3],
#     14:[4,4],
#     15:[5,5]}
# CLASS_IDV = [1,2,3,4,5]

# data_mix = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# dataY = np.asarray(train_label_individual)
# dataX = train_dataX_individual
# outputY = []

# for num in range(len(data_mix)):
#     data_stride = int(data_mix[num] * 128)
#     for i in range(1000):
#         if i == 0:
#             y1 = random.sample(CLASS_IDV, 1)
#         idxA = random.choice(np.argwhere(dataY == y1)[:, 0]) #CC
#         y2 = random.sample(CLASS_IDV, 1)

#         idxA = random.choice(np.argwhere(dataY == y1)[:, 0])
#         while sum([y1,y2],[]) not in PAIR.values():
#             y2 = random.sample(CLASS_IDV, 1)
#         idxB = random.choice(np.argwhere(dataY == y2)[:, 0])
        
#         # test dataset directly use np.concatenate
#         data = np.concatenate((dataX[idxA, :, data_stride:], dataX[idxB, :, :data_stride]), axis=-1)

#         data = np.reshape(data,(1,9,128))

#         label = list(filter(lambda k: PAIR[k] == sum([y1, y2], []), PAIR))
#         y1 = y2
#         idxA = idxB

#         outputY.append(label)
#         if i == 0:
#             outputX = data
#         else:
#             outputX = np.concatenate((outputX, data))

#     print(len(outputY))
#     print(outputX.shape)
#     # print("save test set finished")
#     np.savez("Test_rate\\"+"rate_"+str(data_mix[num])+".npz", outputX, outputY)


####################################################### get sum of body_acc ####################################################

def vector_sum(column):
    sum = 0
    for i in range(0, len(column)):
        sum = sum + column[i] ** 2
    sum = sum ** 0.5
    return sum


#### exp ####
list_SumAcc = []
for i in range(0, len(exp_dataX_list)):
    exp2_sum = np.apply_along_axis(vector_sum, axis=1, arr=exp_dataX_list[i][:,0:3,:])
    list_SumAcc.append(exp2_sum)

#### train #####
train_sumAcc = np.apply_along_axis(vector_sum, axis=1, arr=train_dataX[:,0:3,:])

########################################################### get std ###########################################################

#### exp #####
list_std_exp = []
for i in range(0, len(list_SumAcc)):
    list_std1 = []
    for j in range(0, list_SumAcc[i].shape[0]):
        std = np.std(list_SumAcc[i][j,:])
        list_std1.append(std)
    # print('std:', len(list_std1))
    list_std_exp.append(list_std1)

#### train ######

list_std_train = []
for j in range(0, train_sumAcc.shape[0]):
    std = np.std(train_sumAcc[j,:])
    list_std_train.append(std)


######## to obtain std of walk, stand, up, stand individually ##############
walk_std=[]
stand_std=[]
up_std=[]
down_std=[]
for i in range(0, len(list_std_train)):
    if (train_label[i] == 13):
        walk_std.append(list_std_train[i])
        # print('std of exp walking:', list_std_exp[2][i])
    elif (train_label[i] == 14):
        stand_std.append(list_std_train[i])
        # print('std of exp standing:', list_std_exp[2][i])   
    elif (train_label[i] == 15):
        up_std.append(list_std_train[i])
        # print('std of exp up:', list_std_exp[2][i])
    elif (train_label[i] == 16):
        down_std.append(list_std_train[i])

x1 = range(0,len(walk_std))
x2 = range(0,len(stand_std))
x3 = range(0,len(up_std))
x4 = range(0,len(down_std))


print('walk:', len(walk_std))
print('stand:', len(stand_std))
print('up:', len(up_std))
print('down:', len(down_std))

############################## plot std of each action ######################
# plt.plot(x1, walk_std, 'r d',label = "walking")
# plt.plot(x2, stand_std, 'b 1',label = 'standing')
# plt.plot(x3, up_std, 'g s', label = 'upstairs')
# plt.plot(x4, down_std, 'y *',label = 'downstairs')
# plt.legend()
# plt.grid()
# plt.show()

#################################################### predict by std ###########################################################

def predict_std(list_std):
    predict_list = []
    for i in range(0, len(list_std)):
        std = list_std[i]
        if (std < 0.4):
            predict_list.append(14)
            #predict_list.append(2)
        elif (std>1):
            predict_list.append(15)
            #predict_list.append(3)
        else:
            predict_list.append(13)
            #predict_list.append(1)
        
    return predict_list

#### exp #####
predict_exp_list = []
for i in range(0, len(list_std_exp)):
    predict = predict_std(list_std_exp[i])
    predict_exp_list.append(predict)


##### train ######
predict_train_list = predict_std(list_std_train)
    

###################################################### compare with real label ################################################

print('exp2 label predict:', predict_exp_list[2])
right_num = 0
total_num = len(exp_label_list[2])
print('real:', exp_label_list[2])
print('predict:', predict_exp_list[2])
for j in range(0, len(exp_label_list[2])):
    print('real: {} vs predict: {}'.format(exp_label_list[2][j], predict_exp_list[2][j]))
    if (exp_label_list[2][j] == predict_exp_list[2][j]):
        right_num = right_num + 1
    if (exp_label_list[2][j] == 13 and predict_exp_list[2][j] == 17):
        right_num = right_num + 1
rate = right_num/total_num
print("right rate of exp {}:{} %".format(i, rate*100)) ##### exp 7:34.65346534653465 %

x = range(0, len(predict_exp_list[2]))
y1 = predict_exp_list[2]
y2 = exp_label_list[2]

plt.plot(x, y1)
plt.plot(x, y2)
plt.show()


for i in range(0, len(exp_label_list)):
    print('------------Exp_{}--------'.format(i))
    right_num = 0
    total_num = len(exp_label_list[i])
    print('real:', exp_label_list[i])
    print('predict:', predict_exp_list[i])
    for j in range(0, len(exp_label_list[i])):
        print('real: {} vs predict: {}'.format(exp_label_list[i][j], predict_exp_list[i][j]))
        if (exp_label_list[i][j] == predict_exp_list[i][j]):
            right_num = right_num + 1
        
    rate = right_num/total_num
#     print("right rate of exp {}:{} %".format(i, rate*100))



print("exp 1 std predicted: ")
print(predict_exp_list[1])
print("exp1 length:", len(predict_exp_list[1]))
# for i in range(0, len(exp_label_list[1])):
#     print(exp_label_list[1][i])


###### filter label 13, 14, 15, 16 for training #################################

# train_label_filtered = []
# predict_label_filtered = []


# train_right_num = 0
# train_total_num = 0
# for i in range(0, len(train_label)):
#     if (train_label[i]==13 or train_label[i]==14 or train_label[i]==15 or train_label[i] == 16):
#         train_total_num = train_total_num + 1
#         if (train_label[i] == predict_train_list[i]):
#             train_right_num = train_right_num + 1
#         elif (train_label[i] == 16 and predict_train_list[i]==15):
#             train_right_num = train_right_num + 1

# print("train total num:{}".format(train_total_num))
# print("correct num:{}".format(train_right_num))

# train_rate = train_right_num/train_total_num

# print("correct rate:{}".format(train_rate*100))



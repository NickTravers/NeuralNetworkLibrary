"""
BSD 3-Clause License

Copyright (c) 2017, Remi Cadene
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, *args):
        super(LambdaBase, self).__init__(*args)

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def __init__(self, *args):
        super(Lambda, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def __init__(self, *args):
        super(LambdaMap, self).__init__(*args)
        self.lambda_func = identity

    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def __init__(self, *args):
        super(LambdaReduce, self).__init__(*args)
        self.lambda_func = add

    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def identity(x): return x

def add(x, y): return x + y

resnext101_64x4d_features = nn.Sequential(#Sequential,
    nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3), 1, 1, bias = False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(64, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(256, 256, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(256),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(256, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(256, 512, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(512, 512, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(512),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(512, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(512, 1024, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(1024, 1024, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(1024),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    ),
    nn.Sequential(#Sequential,
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(1024, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (2, 2), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                nn.Sequential(#Sequential,
                    nn.Conv2d(1024, 2048, (1, 1), (2, 2), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
        nn.Sequential(#Sequential,
            LambdaMap( #ConcatTable,
                nn.Sequential(#Sequential,
                    nn.Sequential(#Sequential,
                        nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.Conv2d(2048, 2048, (3, 3), (1, 1), (1, 1), 1, 64, bias = False),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                    ),
                    nn.Conv2d(2048, 2048, (1, 1), (1, 1), (0, 0), 1, 1, bias = False),
                    nn.BatchNorm2d(2048),
                ),
                Lambda(), #Identity,
            ),
            LambdaReduce(), #CAddTable,
            nn.ReLU(),
        ),
    )
)
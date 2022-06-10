#!/bin/bash

source activate reevent

nohup tensorboard --host 0.0.0.0 --logdir runs &
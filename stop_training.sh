#!/bin/bash
pgrep -f train | xargs kill -9
ps -ef | grep train

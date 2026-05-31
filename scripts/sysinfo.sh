#!/bin/bash
#BSUB -J sysinfo
#BSUB -o logs/sysinfo_%J.out
#BSUB -e logs/sysinfo_%J.err
#BSUB -q gpua100
#BSUB -n 1
#BSUB -W 5

echo "=== Linux Version ==="
uname -a

echo ""
echo "=== OS Release ==="
cat /etc/os-release

echo ""
echo "=== CPU Info ==="
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|Cache"

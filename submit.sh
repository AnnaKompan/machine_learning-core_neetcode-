#!/bin/bash
message="Solve $1"
git add .
git commit -m "$message"
git push origin main

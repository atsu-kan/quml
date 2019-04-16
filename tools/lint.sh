while read line
do
    echo $(pwd)/$line
done < <(mypy -p src ; find . -name 'src' -prune -prune -or -name '*.py' -print | xargs mypy)
#find . -name '*.py' -print | xargs pylint
date

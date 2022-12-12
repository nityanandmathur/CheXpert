# Commands for clean text in bash terminal
echo "git_branch() { 
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/' 
}" >> /root/.bashrc 
echo 'export PS1="\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u:\[\033[01;34m\]\w\[\033[00m\] \$(git_branch)\$ "' >> /root/.bashrc

apt update
# jupyter lab --no-browser --port=8890 --allow-root --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password=''
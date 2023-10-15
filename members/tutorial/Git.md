# Git tutorial for this project

## Configure

```bash
$git config --global user.name 'Elaine, Xiaohan ZHONG'
$git config --global user.name 'zxhelaine@gmail.com'
```

## Create your own branch

```bash
# First, clone our git repository to your local machine
$git clone https://github.com/ElaineXHZhong/Transformer-based-Recommendation-Systems-for-Sequential-Recommendation

# Second, create a new branch on your local machine, and switch to it
$git branch Elaine
$git checkout Elaine
# Or, you can use the following command to execute the above commands simultaneously
$git checkout -b Elaine

# Third, push the new branch you just created to github
$git push origin Elaine


# Some help Git commands for you to use about this task
$git branch                 # list all the local branches on local
$git branch -r              # list all the local branches in your Git repository
$git branch -a              # list all the local branches both on local and in your Git repository
$git branch -d Elaine       # delete the local branch named "Elaine"
$git push origin :Elaine    # delete the remote branch named "Elaine" from the remote repository
```

## Pull commits in main branch to your own branch

```bash
# First, switch to main branch
$git checkout main

# Second, pull commits in main branch
$git pull

# Third, switch to your own branch
$git checkout Elaine

# Fourth, merge main with your own branch
$git merge main

# Fifth, Push your own branch to remote repository (!!Note: You should confirm the running is successfully between step 4 and step 5 before your push!)
$git push origin Elaine
```

## Pull commits in your own branch to main branch

```bash
# First, switch to your own branch
$git checkout Elaine

# Second, pull commits in your own branch
$git pull

# Third, switch to main branch
$git checkout main

# Fourth, merge your own branch with main
$git merge Elaine

# Fifth, Push your own branch to remote repository (!!Note: You should confirm the running is successfully between step 4 and step 5 before your push!)
$git push origin main
```
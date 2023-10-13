# Current state: 

**Remote ** / ~~Local~~ 

# Dev histo-miner both locally and remotely

So far i think I will stay only on one branch (to avoid to do a lot of pull request) and pull from remote each time I switch between local and remote! 

While pulling maybe specify:

```bash
git pull                # Update the main repository
git submodule update    # Update submodules
```

To clone:

```bash
git clone --recurse-submodules git@github.com:lucas-sancere/histo-miner.git
```


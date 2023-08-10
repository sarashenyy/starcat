all: install clean

install:
	pip install .

clean:
	rm -rf build starcat.egg-info

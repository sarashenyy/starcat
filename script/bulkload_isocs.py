from starcat import Isoc, Parsec

# use Parsec model
p = Parsec()
i = Isoc(p)

# bulk load isochrones
logage_grid = (6.6, 10, 0.01)
mh_grid = (-0.9, 0.7, 0.01)
i.bulk_load(
    photsyn='gaiaDR2',
    logage_grid=logage_grid, mh_grid=mh_grid
)

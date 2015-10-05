"""
Analyzes euclidean distance between images and shifted images.
"""

import sys
import matplotlib as mpl

sys.path.append('./code')
mpl.use('Agg')

from argparse import ArgumentParser
from cifar import load, vec2patch, patch2vec
from numpy import argmin, sum, square, asarray, unique, mean, median
from numpy.linalg import norm
from numpy.random import rand, permutation
from collections import defaultdict
from pgf.colormap import colormaps
from random import sample
from sklearn.neighbors import KDTree
from scipy.stats import beta
import pgf as pgf
import matplotlib.pyplot as plt

def rank(image, images, i):
	"""
	Returns the number of images closer to `image` than image `i`.
	"""

	d1 = sum(square(image - images[:, [i]]))
	d2 = sum(square(image - images), 0)
	return sum(d2 < d1)


def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--crop', '-C', type=int, default=28)
	parser.add_argument('--num_images', '-N', type=int, default=20)

	args = parser.parse_args(argv[1:])

	if not args.crop < 32:
		print 'args.crop size needs to be smaller than 32.'
		return 1

	images = load([1, 2, 3, 4, 5])[0]
	images = images + rand(*images.shape)
	images = vec2patch(images)

	indices = sample(xrange(images.shape[0]), args.num_images)

	crops = images[:, :args.crop, :args.crop]
	crops = asarray(patch2vec(crops), order='F')

	kdtree = KDTree(crops.T)

	plt.figure(figsize=(5, 10))

	shifts = range(1, 32 - args.crop + 1)
	precision = [1.]

	for shift in shifts:
		print shift

		# shifted images
		images_s = images[indices, shift:shift + args.crop, shift:shift + args.crop]
		images_s = patch2vec(images_s)

		# find nearest neighbors
		_, nbs = kdtree.query(images_s.T)

		precision.append(mean(nbs.ravel() == indices))

	pgf.figure(sans_serif=True)

	precision = asarray(precision)
	prec_mean = precision * 100.
	a = 1. + args.num_images * precision
	b = 1. + args.num_images * (1. - precision)
	prec_95 = beta.ppf(.95, a=a, b=b) * 100.
	prec_05 = beta.ppf(.05, a=a, b=b) * 100.

	pgf.plot([0] + shifts, prec_mean, 'k', line_width=2)
	pgf.plot([0] + shifts, prec_05, 'k--', line_width=1)
	pgf.plot([0] + shifts, prec_95, 'k--', line_width=1)

	pgf.xtick(range(32 - args.crop + 1))
	pgf.xlabel('Shift [pixels]')
	pgf.ylabel('Precision [%]')
	pgf.axis(
		width=4,
		height=4,
		xmin=0,
		xmax=32 - args.crop,
		ymin=0,
		ymax=100,
		label_font_size=8,
		tick_label_font_size=8)
	pgf.box('off')

	pgf.savefig('figures/precision.tex')
	pgf.savefig('figures/precision.pdf')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))

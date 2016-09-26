"""
Analyzes Euclidean distance between images and shifted images.
"""

import sys
import matplotlib as mpl

sys.path.append('./code')
mpl.use('Agg')

from argparse import ArgumentParser
from cifar import load, vec2patch, patch2vec
from numpy import argmin, sum, square, asarray, unique
from numpy.linalg import norm
from numpy.random import rand, permutation
from collections import defaultdict
from pgf.colormap import colormaps
import pgf as pgf
import matplotlib.pyplot as plt

def random_select(k, n):
	return permutation(n)[:k]


def nnb(image, images):
	return argmin(sum(square(images - image), 0))


def main(argv):
	parser = ArgumentParser(argv[0], description=__doc__)
	parser.add_argument('--crop', '-C', type=int, default=28)
	parser.add_argument('--image_id', '-i', type=int, default=13)
	parser.add_argument('--dpi', '-d', type=int, default=150)

	args = parser.parse_args(argv[1:])

	if not args.crop < 32:
		print('args.crop size needs to be smaller than 32.')
		return 1

	images = load([1, 2, 3, 4, 5])[0]
	images = images + rand(*images.shape)
	images = vec2patch(images)

	crops = images[:, :args.crop, :args.crop]
	crops = patch2vec(crops)


	### visualize nearest neighbors

	nnbs = [args.image_id]

	plt.figure(figsize=(5, 10))

	for shift in range(1, 32 - args.crop + 1):
		image = images[[args.image_id], shift:shift + args.crop, shift:shift + args.crop]
		image = patch2vec(image)

		j = nnb(image, crops)

		nnbs.append(j)

		plt.subplot(32 - args.crop, 2, 2 * shift - 1)
		plt.imshow(
			asarray(image.reshape(28, 28, 3), dtype='uint8'),
			interpolation='nearest')
		plt.axis('off')

		plt.subplot(32 - args.crop, 2, 2 * shift)
		plt.imshow(
			asarray(crops[:, j].reshape(28, 28, 3), dtype='uint8'),
			interpolation='nearest')
		plt.axis('off')

	plt.savefig('figures/shifts_{0}_images.png'.format(args.image_id), dpi=args.dpi)


	### visualize Euclidean distances

	nnbs = unique(nnbs)
	nnbs_dist = defaultdict(lambda: [])

	rnds = random_select(100, images.shape[0])
	rnds_dist = defaultdict(lambda: [])

	pgf.figure(sans_serif=True)

	shifts = range(32 - args.crop + 1)

	# compute Euclidean distances
	for shift in shifts:
		image = images[[args.image_id], shift:shift + args.crop, shift:shift + args.crop]
		for j in nnbs:
			nnbs_dist[j].append(norm(image.ravel() - crops[:, j]))
		for j in rnds:
			rnds_dist[j].append(norm(image.ravel() - crops[:, j]))
			
	for j in rnds:
		pgf.plot(shifts, rnds_dist[j], 'k-', opacity=.05, line_width=1)

	cm = colormaps['jet']
	colors = [cm.colors[180], cm.colors[80], cm.colors[220], cm.colors[120]]
	colors = [pgf.RGB(*c) for c in colors]

	for k, j in enumerate(nnbs):
		if j != args.image_id:
			pgf.plot(shifts, nnbs_dist[j], color=colors[k % len(colors)], line_width=2)

	pgf.plot(shifts, nnbs_dist[args.image_id], 'k-', line_width=2)
	pgf.xtick(shifts)
	pgf.xlabel('Shift [pixels]')
	pgf.ylabel('Euclidean distance')
	pgf.axis(
		width=4,
		height=4,
		xmin=0,
		xmax=32 - args.crop,
		ymin=0,
		label_font_size=8,
		tick_label_font_size=8)
	pgf.box('off')

	pgf.savefig('figures/shifts_{0}_distances.tex'.format(args.image_id))
	pgf.savefig('figures/shifts_{0}_distances.pdf'.format(args.image_id))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))

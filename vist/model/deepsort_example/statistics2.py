"""This script is for visualisation plot and do logarithmic fitting."""
# type: ignore
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy

Categories = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
with open("/home/yinjiang/systm/data/motion_cov_Q.pickle", "rb") as handle:
    motion_cov_Q = pickle.load(handle)
with open("/home/yinjiang/systm/data/position_dict.pickle", "rb") as handle:
    position_dict = pickle.load(handle)
with open("/home/yinjiang/systm/data/velocity_dict.pickle", "rb") as handle:
    velocity_dict = pickle.load(handle)
with open("/home/yinjiang/systm/data/det_cov_R.pickle", "rb") as handle:
    detect_cov_R = pickle.load(handle)

for cat in Categories:
    print("-" * 100)
    print(cat)
    Q_mean = np.mean(motion_cov_Q[cat]["deviation"], axis=0)
    Q_cov = np.cov(motion_cov_Q[cat]["deviation"], rowvar=False)
    P0_cov_pos = np.cov(position_dict[cat], rowvar=False)
    P0_cov_vel = np.cov(velocity_dict[cat], rowvar=False)
    print("Q_mean: \n", Q_mean)
    print("Q_cov: \n", Q_cov)
    print("P0_cov_pos: \n", P0_cov_pos)
    print("P0_cov_vel: \n", P0_cov_vel)
    print("-" * 100)

# There is no train category in the BDD100k val dataset,
# two minor categories are motorcycle and rider
visualize_dir = "/home/yinjiang/systm/visualization/statistics"

# for cat in Categories:
#     if os.path.exists(visualize_dir + "/" + cat):
#         shutil.rmtree(visualize_dir + "/" + cat)
#     os.mkdir(visualize_dir + "/" + cat)
# for cat in tqdm(Categories):
#     for i, l in enumerate(["x", "y", "a", "h"]):
#         plt.figure()
#         plt.scatter(
#             motion_cov_Q[cat]["height"],
#             motion_cov_Q[cat]["deviation"][:, i],
#         )
#         plt.xlabel("height")
#         plt.ylabel("deviation " + l)
#         plt.title(cat)
#         plt.savefig(visualize_dir + "/" + cat + "/" + l + "relation" + ".png")
#         plt.clf()

#         plt.figure()
#         mu, std = norm.fit(motion_cov_Q[cat]["deviation"][:, i])
#         plt.hist(motion_cov_Q[cat]["deviation"][:, i], bins=100, density=True)
#         xmin, xmax = plt.xlim()
#         x = np.linspace(xmin, xmax, 100)
#         p = norm.pdf(x, mu, std)
#         plt.plot(x, p, "k", linewidth=2, label=f"mu={mu},std={std}")
#         plt.xlabel("deviation " + l)
#         plt.ylabel("density")
#         plt.savefig(visualize_dir + "/" + cat + "/" + l + "gaussian" + ".png")
#         plt.clf()
'''
for cat in Categories:
    mini_group = []
    heights = motion_cov_Q[cat]["height"]
    aspect_ratio = motion_cov_Q[cat]["deviation"][:, 2]
    # print("length of values: ", len(heights))

    h_a = list(zip(heights, aspect_ratio))
    sorted_ha = sorted(h_a, key=lambda x: x[0])
    i = 0
    while i < len(sorted_ha):
        mini_group.append(sorted_ha[i : i + 100])
        i += 100

    cov_a = []
    mean_h = []
    for group in mini_group:
        h_mini, a_mini = zip(*group)
        cov_a.append(np.cov(a_mini))
        mean_h.append(np.mean(h_mini))

    # plt.figure()
    # plt.scatter(
    #     mean_h,
    #     cov_a,
    # )
    # plt.xlabel("height")
    # plt.ylabel("a ")
    # plt.title(cat)
    # # plt.savefig(visualize_dir + "/" + cat + "/" + l + "relation" + ".png")
    # plt.show()
    # plt.clf()

    def func_log(x, a, b, c):
        """Return values from a general log function."""
        return a * np.log(b * x) + c

    mean_h = np.array(mean_h)
    cov_a = np.array(cov_a)

    # perform the fit
    params, cv = scipy.optimize.curve_fit(func_log, mean_h, cov_a)
    m, t, b = params
    sampleRate = 20_000  # Hz
    tauSec = (1 / t) / sampleRate

    # determine quality of the fit
    squaredDiffs = np.square(cov_a - func_log(mean_h, m, t, b))
    squaredDiffsFromMean = np.square(cov_a - np.mean(cov_a))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    # plot the results
    plt.figure()
    plt.plot(mean_h, cov_a, ".", label="data")
    plt.plot(mean_h, func_log(mean_h, m, t, b), "--", label="fitted")
    plt.title("Fitted logarithmic Curve")
    plt.savefig(visualize_dir + "/" + cat + "/" + "a_" + "fitting" + ".png")
    plt.clf()

    # inspect the parameters
    print("#" * 100)
    print(cat)
    # print(f"Y = {m} * log({t} * x) + {b}")
    print(f"m ={m}, t={t}, b={b}")
    print(f"Tau = {tauSec * 1e6} µs")
    print("#" * 100)
'''
print("done")

# for i in range(4):
#     plt.scatter(
#         detect_cov_R["car"]["detections"][:, 3],
#         detect_cov_R["car"]["deviation"][:, i],
#     )
#     plt.xlabel("height")
#     plt.ylabel(f"deviation {i}")
#     plt.title("car")
#     plt.show()

#     mu, std = norm.fit(detect_cov_R["car"]["deviation"][:, i])
#     plt.hist(detect_cov_R["car"]["deviation"][:, i], bins=100, density=True)
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, std)
#     plt.plot(x, p, "k", linewidth=2, label=f"mu={mu},std={std}")
#     plt.xlabel(f"deviation {i}")
#     plt.title("car")
#     plt.show()
# for i in range(4):
#     plt.scatter(
#         detect_cov_R["pedestrian"]["detections"][:, 3],
#         detect_cov_R["pedestrian"]["deviation"][:, i],
#     )
#     plt.xlabel("height")
#     plt.ylabel(f"deviation {i}")
#     plt.title("pedestrian")
#     plt.show()

#     mu, std = norm.fit(detect_cov_R["pedestrian"]["deviation"][:, i])
#     plt.hist(
#         detect_cov_R["pedestrian"]["deviation"][:, i], bins=100, density=True
#     )
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, std)
#     plt.plot(x, p, "k", linewidth=2, label=f"mu={mu},std={std}")
#     plt.xlabel(f"deviation {i}")
#     plt.title("pedestrian")
#     plt.show()

print("#" * 100)
for cat in Categories:
    print("-" * 100)
    print(cat)
    R_mean = np.mean(detect_cov_R[cat]["deviation"], axis=0)
    R_cov = np.cov(detect_cov_R[cat]["deviation"], rowvar=False)
    print("R_mean: \n", R_mean)
    print("R_cov: \n", R_cov)
    print("-" * 100)


# There is no train category in the BDD100k val dataset,
# two minor categories are motorcycle and rider
visualize_dir = "/home/yinjiang/systm/visualization/statistics"

# for cat in Categories:
#     if os.path.exists(visualize_dir + "/" + cat):
#         shutil.rmtree(visualize_dir + "/" + cat)
#     os.mkdir(visualize_dir + "/" + cat)
# for cat in tqdm(Categories):
#     for i, l in enumerate(["x", "y", "a", "h"]):
#         plt.figure()
#         plt.scatter(
#             motion_cov_Q[cat]["height"],
#             motion_cov_Q[cat]["deviation"][:, i],
#         )
#         plt.xlabel("height")
#         plt.ylabel("deviation " + l)
#         plt.title(cat)
#         plt.savefig(visualize_dir + "/" + cat + "/" + l + "relation" + ".png")
#         plt.clf()

#         plt.figure()
#         mu, std = norm.fit(motion_cov_Q[cat]["deviation"][:, i])
#         plt.hist(motion_cov_Q[cat]["deviation"][:, i], bins=100, density=True)
#         xmin, xmax = plt.xlim()
#         x = np.linspace(xmin, xmax, 100)
#         p = norm.pdf(x, mu, std)
#         plt.plot(x, p, "k", linewidth=2, label=f"mu={mu},std={std}")
#         plt.xlabel("deviation " + l)
#         plt.ylabel("density")
#         plt.savefig(visualize_dir + "/" + cat + "/" + l + "gaussian" + ".png")
#         plt.clf()

'''
for cat in Categories:
    mini_group = []
    heights = detect_cov_R[cat]["detections"][:, 3]
    aspect_ratio = detect_cov_R[cat]["deviation"][:, 2]
    # print("length of values: ", len(heights))

    h_a = list(zip(heights, aspect_ratio))
    sorted_ha = sorted(h_a, key=lambda x: x[0])
    i = 0
    while i < len(sorted_ha):
        mini_group.append(sorted_ha[i : i + 100])
        i += 100

    cov_a = []
    mean_h = []
    for group in mini_group:
        h_mini, a_mini = zip(*group)
        cov_a.append(np.cov(a_mini))
        mean_h.append(np.mean(h_mini))

    # plt.figure()
    # plt.scatter(
    #     mean_h,
    #     cov_a,
    # )
    # plt.xlabel("height")
    # plt.ylabel("a ")
    # plt.title(cat)
    # # plt.savefig(visualize_dir + "/" + cat + "/" + l + "relation" + ".png")
    # plt.show()
    # plt.clf()

    def func_log(x, a, b, c):
        """Return values from a general log function."""
        return a * np.log(b * x) + c

    mean_h = np.array(mean_h)
    cov_a = np.array(cov_a)

    # perform the fit
    params, cv = scipy.optimize.curve_fit(func_log, mean_h, cov_a)
    m, t, b = params
    sampleRate = 20_000  # Hz
    tauSec = (1 / t) / sampleRate

    # determine quality of the fit
    squaredDiffs = np.square(cov_a - func_log(mean_h, m, t, b))
    squaredDiffsFromMean = np.square(cov_a - np.mean(cov_a))
    rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    print(f"R² = {rSquared}")

    # plot the results
    # plt.plot(mean_h, cov_a, ".", label="data")
    # plt.plot(mean_h, func_log(mean_h, m, t, b), "--", label="fitted")
    # plt.title("Fitted Exponential Curve")

    # inspect the parameters
    print("#" * 100)
    print(cat)
    # print(f"Y = {m} * log({t} * x) + {b}")
    print(f"m ={m}, t={t}, b={b}")
    print(f"Tau = {tauSec * 1e6} µs")
    print("#" * 100)

'''

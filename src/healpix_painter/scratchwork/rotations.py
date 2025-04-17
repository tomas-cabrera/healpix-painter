import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(1245)

# Yeah yeah it's not uniform over the sphere but whatever
ref_dec = rng.uniform(-90, 90) * np.pi / 180
ref_ra = rng.uniform(0, 360) * np.pi / 180
ref_phi = ref_ra
print(f"ref_phi: {ref_phi:.3f} rad ({ref_phi*180/np.pi:.3f} deg)")
deltara = 2 * 15 * np.pi / 180
deltadec = 90 * np.pi / 180

sinphi, cosphi = np.sin(ref_phi), np.cos(ref_phi)
sinra, cosra = np.sin(deltara), np.cos(deltara)

rms = []
steps = 20
for i in range(1, steps + 1):
    stepdec = deltadec * i / steps
    print(f"stepdec: {stepdec:.3f} rad ({stepdec*180/np.pi:.3f} deg)")
    sindec, cosdec = np.sin(stepdec), np.cos(stepdec)
    dec_rotation_matrix = (
        np.array([[cosphi, -sinphi, 0], [sinphi, cosphi, 0], [0, 0, 1]])
        @ np.array([[cosdec, 0, -sindec], [0, 1, 0], [sindec, 0, cosdec]])
        @ np.array([[cosphi, sinphi, 0], [-sinphi, cosphi, 0], [0, 0, 1]])
    )
    rms.append(dec_rotation_matrix)

ax = plt.figure().add_subplot(projection="3d")
ax.axes.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
ax.plot([-1, 1], [0, 0], [0, 0], "r--")
ax.plot([0, 0], [-1, 1], [0, 0], "g--")
ax.plot([0, 0], [0, 0], [-1, 1], "b--")

sinrr, cosrr = np.sin(ref_ra), np.cos(ref_ra)
sindr, cosdr = np.sin(ref_dec), np.cos(ref_dec)
x = np.array([cosrr * cosdr, sinrr * cosdr, sindr])
plotx = [[x[0]], [x[1]], [x[2]]]
print(f"x: {x}")
for rm in rms:
    x_rotated = rm @ x
    print(f"rotated x: {x_rotated}")
    plotx[0].append(x_rotated[0])
    plotx[1].append(x_rotated[1])
    plotx[2].append(x_rotated[2])
ax.plot(
    *plotx,
    "-",
    color="k",
    lw=2,
)

for i in range(10):
    ra_sample = rng.uniform(0, 2 * np.pi)
    dec_sample = rng.uniform(-np.pi / 2, np.pi / 2)
    sinrs, cosrs = np.sin(ra_sample), np.cos(ra_sample)
    sinds, cosds = np.sin(dec_sample), np.cos(dec_sample)

    x = np.array([cosrs * cosds, sinrs * cosds, sinds])
    plotx = [[x[0]], [x[1]], [x[2]]]
    for rm in rms:
        x_rotated = rm @ x
        plotx[0].append(x_rotated[0])
        plotx[1].append(x_rotated[1])
        plotx[2].append(x_rotated[2])

    ax.plot(
        *plotx,
        "-",
    )
plt.show()
plt.close()

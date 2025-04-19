import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

rng = np.random.default_rng(123456)

# Yeah yeah it's not uniform over the sphere but whatever
ref_dec = jnp.array(rng.uniform(-90, 90) * jnp.pi / 180)
ref_ra = jnp.array(rng.uniform(0, 360) * jnp.pi / 180)
ref_phi = ref_ra
print(f"ref_phi: {ref_phi:.3f} rad ({ref_phi * 180 / jnp.pi:.3f} deg)")
deltara = 2 * 15 * jnp.pi / 180
deltadec = 90 * jnp.pi / 180

sinphi, cosphi = jnp.sin(ref_phi), jnp.cos(ref_phi)
sinra, cosra = jnp.sin(deltara), jnp.cos(deltara)

rms = []
steps = 20
for i in range(1, steps + 1):
    stepdec = deltadec * i / steps
    print(f"stepdec: {stepdec:.3f} rad ({stepdec * 180 / jnp.pi:.3f} deg)")
    sindec, cosdec = jnp.sin(stepdec), jnp.cos(stepdec)
    dec_rotation_matrix = jnp.linalg.multi_dot(
        [
            jnp.array([[cosphi, -sinphi, 0], [sinphi, cosphi, 0], [0, 0, 1]]),
            jnp.array([[cosdec, 0, -sindec], [0, 1, 0], [sindec, 0, cosdec]]),
            jnp.array([[cosphi, sinphi, 0], [-sinphi, cosphi, 0], [0, 0, 1]]),
        ]
    )
    rms.append(dec_rotation_matrix)

ax = plt.figure().add_subplot(projection="3d")
ax.axes.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
ax.plot([-1, 1], [0, 0], [0, 0], "r--")
ax.plot([0, 0], [-1, 1], [0, 0], "g--")
ax.plot([0, 0], [0, 0], [-1, 1], "b--")

sinrr, cosrr = jnp.sin(ref_ra), jnp.cos(ref_ra)
sindr, cosdr = jnp.sin(ref_dec), jnp.cos(ref_dec)
x = jnp.array([cosrr * cosdr, sinrr * cosdr, sindr])
plotx = [[x[0]], [x[1]], [x[2]]]
print(f"x: {x}")
for rm in rms:
    x_rotated = jnp.dot(rm, x)
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
    ra_sample = jnp.array(rng.uniform(0, 2 * jnp.pi))
    dec_sample = jnp.array(rng.uniform(-jnp.pi / 2, jnp.pi / 2))
    sinrs, cosrs = jnp.sin(ra_sample), jnp.cos(ra_sample)
    sinds, cosds = jnp.sin(dec_sample), jnp.cos(dec_sample)

    x = jnp.array([cosrs * cosds, sinrs * cosds, sinds])
    plotx = [[x[0]], [x[1]], [x[2]]]
    for rm in rms:
        x_rotated = jnp.dot(rm, x)
        plotx[0].append(x_rotated[0])
        plotx[1].append(x_rotated[1])
        plotx[2].append(x_rotated[2])

    ax.plot(
        *plotx,
        "-",
    )
plt.show()
plt.close()

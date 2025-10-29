import os
import os.path as pa
import shutil

import matplotlib.pyplot as plt
import numpy as np

from healpix_painter.plotting import FILTER2COLOR


# Output results
def package_results(
    skymap_filename,
    selected_pointings,
    footprint,
    output_dir=None,
):
    # Copy skymap to output directory, or define output as skymap directory
    if output_dir is not None:
        # Make output dir
        os.makedirs(output_dir, exist_ok=True)
        # Copy skymap
        shutil.copy(skymap_filename, output_dir)
    else:
        output_dir = pa.dirname(skymap_filename)

    # Make prob vs n_pointings plot
    plt.figure(figsize=(4, 3))
    for f in selected_pointings.keys():
        plt.plot(
            np.arange(len(selected_pointings[f]["probs_added"])),
            np.cumsum(selected_pointings[f]["probs_added"]),
            label=f"{f} {(np.sum(selected_pointings[f]['probs_added']) * 100):.2f}% covered",
            color=FILTER2COLOR.get(f),
        )
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Number of Pointings")
    plt.ylabel("Cumulative Probability Covered")
    plt.savefig(pa.join(output_dir, "cumprob_npointings.png"))
    plt.savefig(pa.join(output_dir, "cumprob_npointings.pdf"))
    plt.close()

    # Save footprint plots and csvs
    # Iterate over filters
    for f in selected_pointings.keys():
        # Get + save data
        filter_data = selected_pointings[f]
        filter_data.to_csv(pa.join(output_dir, f"pointings_{f}.csv"), index=False)
        # Set pointing limit for plotting
        # TODO: change pointing limit, currently 480 (about 12 hours of 60s DECam exposures)
        n_exp = min(480, len(filter_data["probs_added"]))
        # Calculate alphas and line widths for plotting
        alpha = np.interp(
            filter_data["probs_added"],
            (np.min(filter_data["probs_added"]), np.max(filter_data["probs_added"])),
            (0.2, 1),
        )
        lw = alpha + 1
        # Plot footprints
        # TODO: improve plotting (currently just RA/Dec scatter, needs spherical coords)
        plt.figure(figsize=(6, 6))
        for i in range(n_exp):
            fpc = footprint.rotate(
                filter_data.iloc[i]["ra"], filter_data.iloc[i]["dec"]
            )
            for fpccd in fpc:
                plt.plot(
                    fpccd[0],
                    fpccd[1],
                    color=FILTER2COLOR.get(f),
                    alpha=alpha[i],
                    lw=lw[i],
                )
        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")
        plt.title(
            f"Pointings for filter {f} [{n_exp}/{len(filter_data['probs_added'])}]"
        )
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.tight_layout()
        plt.savefig(pa.join(output_dir, f"footprints_{f}.png"))
        plt.savefig(pa.join(output_dir, f"footprints_{f}.pdf"))
        plt.close()

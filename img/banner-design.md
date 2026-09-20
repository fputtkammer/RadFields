# Banner design — Clear Fields

The banner contains only a title, one subtitle, and an organ-to-field illustration. Open space provides the separation; there are no accent bars, badges, frames around the composition, or background fills. Careful alignment carries the hierarchy.

Anatomy is simplified into flat vector shapes. The schematic uses an anterior view: patient right appears on the viewer’s left. The liver sits predominantly on patient right, the cardiac apex points to patient left, and the right kidney is lower. This is an illustrative schematic, not a diagnostic image.

Subtle transverse guides suggest CT levels. Color-matched connections lead from the lungs, liver, and kidneys into distinct groups of text fields, expressing the organization of report findings by organ. Small repeated organ symbols reinforce the grouping without extra labels.

Typography and anatomy remain vector paths throughout. Restrained strokes and matte fills replace rendering effects. The light and dark variants use the same geometry and transparent canvas, with text colors adjusted for contrast. Legibility and spacing are checked at README display size.

## Assets

- `banner.svg`: light-mode vector master, transparent, 1500 × 420 viewBox.
- `banner-dark.svg`: dark-mode counterpart, also transparent.
- `banner.png` and `banner-dark.png`: transparent 2400 × 672 PNG exports.
- The README selects the appropriate SVG through a `picture` element.
- The final artwork is entirely vector-authored; no generated raster illustration is embedded. Earlier generated concept art was not retained in the repository.
- Typography: Bricolage Grotesque Bold and Instrument Sans Regular (SIL Open Font License), outlined for portable rendering. Readable SVG group labels preserve the wording.

RadFields is the working public-facing name. The GitHub repository URL and Python import name remain unchanged pending the repository-owner rename and the naming decision.

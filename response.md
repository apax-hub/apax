1. Can't we remove parity_debug entirely?
2. easy fix? Implement!
3. easy fix? Implement!
4. Needs further investigation! I don't like filter per name because less future proof! Consider possible solutions / defer?
5. reset_layers for shape-mismatched slots is mechanically derivable from the converter output. This is not a bug / should not happen automatically!! If you introduce e.g. a shallow ensemble the user MUST specify the layers they want to replace thus the ones needed to be reset. Resetting means loosing weights and this should not be automatic!!
6. remove dead if block!
7. fix by extracting
8. Remove the MACE branch from the base class.
9. extract
10. collapse, remove comments!
11. resolve, I don't like `hasattr(x, "value")` calls in the first place, is it necessary?
12. hoist is fine, no need to make private!
13. the conversion functions are utils that might eventually be removed, no need to cleanup.
14. narrow done
15. Lift or fail loudly, you decide
16. fix the bad pattern
17. Investigate further, propose the better fix!
18. why does the loss have inputs: dict = {} as default? Why was this added in this branch?!

Remove the worthless comments.

For the remainder it looks good.
- DROP class alias! All code like `InteractionBlock = InteractionBlockResidual` must be removed! This is a new feature branch, remove all traces from work on that branch that eventually has been removed like this!!

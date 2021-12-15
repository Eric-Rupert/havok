# havok
A module implementing the havok algorithm


60 time steps/chord length
system reconstruction is most successful with a hank_len = 30-40, longer than that and the system diverges
adding orders to Theta beyond 3 or using sin functions does not significantly improve results

Generally this technique is unable to find the wall as a forcing function, rather detects forcing at the beginning of the trajectory.

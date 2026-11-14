# 3. Background and related work

*Author: Patrick. Target about 0.75 page in the body. Literature search and first draft done with AI assistance from Claude; declare this in the AI-use statement. References continue §2's numbering, so these become references 8 to 25 of the merged report. [CHECK before submission: Kurte et al. [19] was read only as an abstract, and the search for 2024 to 2026 carbon-aware RL work was limited. Read Kurte in full and run one more search before keeping the word "none" in §3.5.]*

## 3.1 Thermal mass and carbon accounting

An office's thermal mass, the heat stored in its structure and contents, makes the indoor temperature lag its heat gains by hours. Pre-cooling banks cooling in that mass, but the cooling leaks back through the envelope, and the savings depend on construction, equipment, tariffs and climate [8]. Carbon accounting can use an average factor, total emissions over total generation, or a marginal rate, the change in emissions when demand changes. Marginal rates can differ by more than a factor of two between hours in the same region, and the cheapest hours to generate are often the dirtiest at the margin [9]. Our signal is an average intensity, a limitation §10 returns to. §4 and §6 define the MDP and the algorithms.

## 3.2 Conventional control

Most buildings run on rules, such as a fixed setpoint or an occupancy setback. Rules are cheap and easy to audit but react only to current conditions. Model predictive control optimises actions over a forecast horizon using a model of the building. Oldewurtel et al. found that across 1,280 simulated cases, rule-based control used over 40% more non-renewable primary energy than a predictive controller with perfect knowledge in more than half of them, and that bound assumed occupancy was known [10]. Predictive control needs an accurate model, extra infrastructure and an expert team, and its practical use is still at an early stage [11]. Our oracle is a perfect-knowledge bound of the same kind, and we tune our rules as carefully as the learners.

## 3.3 Reinforcement learning for building control

Only about 11% of the reinforcement learning applications Wang and Hong reviewed ran in real buildings [12]. Early studies often reported large savings against weak baselines [13]. Wei et al., for example, cut energy cost by 20% to 70% against an on/off thermostat fixed at 24 °C, from single runs [14]. Against strong rules the gain shrinks. Lu et al. found that PPO performed about the same as ASHRAE Guideline 36 sequences in a five-zone office [15], and in a benchmark of PPO, SAC and TD3 no agent beat the rule-based controller in the office building [13]. Our own results in §8.1 repeat this pattern. The shared benchmarks that track carbon score it rather than reward it. BOPTEST reports CO₂ from time-varying emission factors [16]. CityLearn v2 lets agents observe carbon intensity, but its controllers cut emissions little, which its authors blame on a weak carbon signal [17]. We built a low-order simulator because it is fast enough for five seeds of every controller and simple enough to solve exactly.

## 3.4 Carbon-aware, multi-objective and constrained control

Few studies reward a learned building controller on carbon. Jeen et al.'s agent, rewarded on energy times carbon intensity plus discomfort, cut emissions by up to 31% against a rule in one of three simulated buildings, and the rule emitted less in the other two [18]. Kurte et al. reported 6% to 16% lower carbon from pre-cooling learned under a marginal emissions signal [19]. Jeen et al. do not separate timing from energy reduction, and Kurte et al.'s abstract reports only the total saving. Studies that isolate timing find it small. Shifting chiller load in a Californian office while holding annual energy roughly constant avoided about 2% of annual emissions [20], and our timing gain of about 1% falls in the same range. Storage sets the ceiling. Predictive control with CO₂ forecasts saved about 16% in well-insulated Danish buildings but 0% to 5% in older ones [21].

A weighted reward like ours reaches only the trade-offs on the convex hull of what is achievable [22], so the comfort-weight sweep in §8.3 approximates the trade-off curve. A constrained MDP instead minimises one cost subject to limits on the others [23]. Lagrangian methods adjust the weight during training until the limit holds, and Ray et al. argue that a fixed weight cannot be chosen in advance to hit a target level [24]. Sanchez and Cai found that, in a single-zone office, treating comfort as a constraint gave cost savings comparable to predictive control, while unconstrained learning gave very different results for different penalty weights [25].

## 3.5 Gap

Among the carbon-aware reinforcement learning studies we reviewed, none compares against both a tuned rule-based controller and a perfect-foresight optimum. None splits the carbon saving into timing and energy reduction, and none measures how much of the achievable saving depends on knowing occupancy rather than carbon intensity. We do all three, on a building simple enough to solve exactly.

## References

8. Braun JE. Load control using building thermal mass. J Sol Energy Eng. 2003;125(3):292–301. doi:10.1115/1.1592184
9. Graff Zivin JS, Kotchen MJ, Mansur ET. Spatial and temporal heterogeneity of marginal emissions: implications for electric cars and other electricity-shifting policies. J Econ Behav Organ. 2014;107(Pt A):248–68. doi:10.1016/j.jebo.2014.03.010
10. Oldewurtel F, Parisio A, Jones CN, Gyalistras D, Gwerder M, Stauch V, et al. Use of model predictive control and weather forecasts for energy efficient building climate control. Energy Build. 2012;45:15–27. doi:10.1016/j.enbuild.2011.09.022
11. Drgoňa J, Arroyo J, Cupeiro Figueroa I, Blum D, Arendt K, Kim D, et al. All you need to know about model predictive control for buildings. Annu Rev Control. 2020;50:190–232. doi:10.1016/j.arcontrol.2020.09.001
12. Wang Z, Hong T. Reinforcement learning for building controls: the opportunities and challenges. Appl Energy. 2020;269:115036. doi:10.1016/j.apenergy.2020.115036
13. Manjavacas A, Campoy-Nieves A, Jiménez-Raboso J, Molina-Solana M, Gómez-Romero J. An experimental evaluation of deep reinforcement learning algorithms for HVAC control. Artif Intell Rev. 2024;57(7):173. doi:10.1007/s10462-024-10819-x
14. Wei T, Wang Y, Zhu Q. Deep reinforcement learning for building HVAC control. In: Proceedings of the 54th ACM/EDAC/IEEE Design Automation Conference (DAC); 2017 Jun 18–22; Austin, TX, USA. New York: ACM; 2017. doi:10.1145/3061639.3062224
15. Lu X, Fu Y, Xu S, Zhu Q, O'Neill Z, Yang Z. Comparison study of high-performance rule-based HVAC control with deep reinforcement learning-based control in a multi-zone VAV system. In: Proceedings of the 7th International High Performance Buildings Conference at Purdue; 2022; West Lafayette, IN, USA. Paper 407. Available from: https://docs.lib.purdue.edu/ihpbc/407
16. Blum D, Arroyo J, Huang S, Drgoňa J, Jorissen F, Walnum HT, et al. Building optimization testing framework (BOPTEST) for simulation-based benchmarking of control strategies in buildings. J Build Perform Simul. 2021;14(5):586–610. doi:10.1080/19401493.2021.1986574
17. Nweye K, Kaspar K, Buscemi G, Fonseca T, Pinto G, Ghose D, et al. CityLearn v2: energy-flexible, resilient, occupant-centric, and carbon-aware management of grid-interactive communities. J Build Perform Simul. 2025;18(1):17–38. doi:10.1080/19401493.2024.2418813
18. Jeen SR, Abate A, Cullen JM. Low emission building control with zero-shot reinforcement learning. Proc AAAI Conf Artif Intell. 2023;37(12):14259–67. doi:10.1609/aaai.v37i12.26668
19. Kurte K, Amasyali K, Munk J, Zandi H. Deep reinforcement learning based HVAC control for reducing carbon footprint of buildings. In: 2023 IEEE Power & Energy Society Innovative Smart Grid Technologies Conference (ISGT); 2023 Jan 16–19; Washington, DC, USA. Piscataway (NJ): IEEE; 2023. p. 1–5. doi:10.1109/ISGT51731.2023.10066358
20. Zou A, Duarte Roa C, Schiavon S. Quantifying office building HVAC marginal operating carbon emissions and load shift potential: a case study in California. In: ASHRAE Winter Conference Proceedings; 2025 Feb. Available from: https://escholarship.org/uc/item/6nx97049
21. Leerbeck K, Bacher P, Junker RG, Tveit A, Corradi O, Madsen H, et al. Control of heat pumps with CO2 emission intensity forecasts. Energies. 2020;13(11):2851. doi:10.3390/en13112851
22. Roijers DM, Vamplew P, Whiteson S, Dazeley R. A survey of multi-objective sequential decision-making. J Artif Intell Res. 2013;48:67–113. doi:10.1613/jair.3987
23. Altman E. Constrained Markov decision processes. Boca Raton (FL): Chapman & Hall/CRC; 1999.
24. Ray A, Achiam J, Amodei D. Benchmarking safe exploration in deep reinforcement learning [Internet]. San Francisco: OpenAI; 2019 [cited 2026 Nov 15]. Available from: https://cdn.openai.com/safexp-short.pdf
25. Sanchez J, Cai J. Building demand response control through constrained reinforcement learning with linear policies. Appl Energy. 2025;398:126404. doi:10.1016/j.apenergy.2025.126404

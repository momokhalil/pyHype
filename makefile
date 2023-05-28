run_dmr:
	mpiexec -n 4 python -m examples.dmr.dmr

run_explosion:
	mpiexec -n 1 python -m examples.explosion.explosion

run_explosion_multi:
	mpiexec -n 4 python -m examples.explosion_multi.explosion

run_implosion:
	mpiexec -n 1 python -m examples.implosion.implosion

run_jet:
	mpiexec -n 4 python -m examples.jet.jet

run_shockbox:
	mpiexec -n 4 python -m examples.shockbox.shockbox

run_supersonic_step:
	mpiexec -n 4 python -m examples.supersonic_step.supersonic_step

run_supersonic_wedge:
	mpiexec -n 2 python -m examples.supersonic_wedge.supersonic_wedge

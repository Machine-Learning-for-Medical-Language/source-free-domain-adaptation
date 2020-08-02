competition/scoring_program.zip: scoring_program/*
	cd scoring_program && zip -FS -r ../competition/scoring_program.zip * && cd ..

competition/practice_data.zip: $(shell find practice_data)
	cd practice_data && zip -FS -r ../competition/practice_data.zip * && cd ..

competition.zip: competition/* competition/scoring_program.zip competition/practice_data.zip
	cd competition && zip ../competition.zip * && cd ..

submission.zip: submission/*
	cd submission && zip -FS -r ../submission.zip * && cd ..
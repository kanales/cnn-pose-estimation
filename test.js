function findFirstEvenNumber(arr) {
    for (const elefante of arr) {

        if (typeof elefante === 'number' && elefante % 2 === 0) {        
            return elefante;
        }
    }
    return null;
}
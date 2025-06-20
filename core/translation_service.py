"""
Translation Service - Core functionality for translating content to English with discipline-specific terminology.

Goal: Provide standardized translation of educational content into English using
appropriate academic terminology for each discipline.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Goal: Translate educational content to English with discipline-specific terminology.
    
    Provides translation capabilities for academic content while maintaining
    proper terminology standards for each discipline.
    """
    
    def __init__(self):
        """Initialize translation service with discipline-specific dictionaries."""
        self.discipline_translations = self._initialize_discipline_translations()
        self.common_academic_terms = self._initialize_common_academic_terms()
        logger.info("TranslationService initialized with %d discipline mappings", 
                   len(self.discipline_translations))
    
    def translate_to_english(self, text: str, source_language: str, discipline: str) -> str:
        """
        Goal: Translate text to English using standard terminology for the discipline.
        
        Args:
            text: Text to translate
            source_language: Source language code (e.g., 'spanish', 'french')
            discipline: Academic discipline for terminology standardization
            
        Returns:
            Translated text with standardized terminology
        """
        if source_language.lower() == 'english':
            return self._standardize_english_terminology(text, discipline)
        
        logger.debug(f"Translating '{text}' from {source_language} to English for {discipline}")
        
        # Get translation mappings
        translations = self._get_translations_for_language(source_language, discipline)
        
        # Apply translations
        translated = self._apply_translations(text, translations)
        
        # Standardize terminology
        standardized = self._standardize_english_terminology(translated, discipline)
        
        logger.debug(f"Translation result: '{standardized}'")
        return standardized
    
    def _initialize_discipline_translations(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Goal: Initialize comprehensive translation dictionaries for academic disciplines.
        
        Structure: {language: {discipline: {foreign_term: english_term}}}
        """
        return {
            'spanish': {
                'Physics': {
                    'física': 'Physics',
                    'mecánica': 'Mechanics',
                    'mecánica clásica': 'Classical Mechanics',
                    'mecánica cuántica': 'Quantum Mechanics',
                    'termodinámica': 'Thermodynamics',
                    'electromagnetismo': 'Electromagnetism',
                    'óptica': 'Optics',
                    'relatividad': 'Relativity',
                    'energía': 'Energy',
                    'fuerza': 'Force',
                    'velocidad': 'Velocity',
                    'aceleración': 'Acceleration',
                    'momento': 'Momentum',
                    'campo eléctrico': 'Electric Field',
                    'campo magnético': 'Magnetic Field',
                    'ondas': 'Waves',
                    'oscilaciones': 'Oscillations',
                    'física nuclear': 'Nuclear Physics',
                    'física atómica': 'Atomic Physics',
                    'astrofísica': 'Astrophysics',
                    'materia condensada': 'Condensed Matter'
                },
                'Mathematics': {
                    'matemáticas': 'Mathematics',
                    'cálculo': 'Calculus',
                    'álgebra': 'Algebra',
                    'geometría': 'Geometry',
                    'trigonometría': 'Trigonometry',
                    'estadística': 'Statistics',
                    'probabilidad': 'Probability',
                    'ecuaciones diferenciales': 'Differential Equations',
                    'álgebra lineal': 'Linear Algebra',
                    'análisis': 'Analysis',
                    'topología': 'Topology',
                    'teoría de números': 'Number Theory'
                },
                'Chemistry': {
                    'química': 'Chemistry',
                    'química orgánica': 'Organic Chemistry',
                    'química inorgánica': 'Inorganic Chemistry',
                    'fisicoquímica': 'Physical Chemistry',
                    'bioquímica': 'Biochemistry',
                    'química analítica': 'Analytical Chemistry',
                    'termodinámica química': 'Chemical Thermodynamics',
                    'cinética química': 'Chemical Kinetics',
                    'equilibrio químico': 'Chemical Equilibrium',
                    'enlaces químicos': 'Chemical Bonds',
                    'reacciones químicas': 'Chemical Reactions'
                },
                'Biology': {
                    'biología': 'Biology',
                    'biología molecular': 'Molecular Biology',
                    'biología celular': 'Cell Biology',
                    'genética': 'Genetics',
                    'evolución': 'Evolution',
                    'ecología': 'Ecology',
                    'fisiología': 'Physiology',
                    'anatomía': 'Anatomy',
                    'microbiología': 'Microbiology',
                    'neurociencia': 'Neuroscience',
                    'biología del desarrollo': 'Developmental Biology',
                    'biotecnología': 'Biotechnology'
                }
            },
            'french': {
                'Physics': {
                    'physique': 'Physics',
                    'mécanique': 'Mechanics',
                    'mécanique classique': 'Classical Mechanics',
                    'mécanique quantique': 'Quantum Mechanics',
                    'thermodynamique': 'Thermodynamics',
                    'électromagnétisme': 'Electromagnetism',
                    'optique': 'Optics',
                    'relativité': 'Relativity',
                    'énergie': 'Energy',
                    'force': 'Force',
                    'vitesse': 'Velocity',
                    'accélération': 'Acceleration',
                    'quantité de mouvement': 'Momentum',
                    'champ électrique': 'Electric Field',
                    'champ magnétique': 'Magnetic Field',
                    'ondes': 'Waves',
                    'oscillations': 'Oscillations',
                    'physique nucléaire': 'Nuclear Physics',
                    'physique atomique': 'Atomic Physics',
                    'astrophysique': 'Astrophysics',
                    'matière condensée': 'Condensed Matter'
                },
                'Mathematics': {
                    'mathématiques': 'Mathematics',
                    'calcul': 'Calculus',
                    'algèbre': 'Algebra',
                    'géométrie': 'Geometry',
                    'trigonométrie': 'Trigonometry',
                    'statistiques': 'Statistics',
                    'probabilités': 'Probability',
                    'équations différentielles': 'Differential Equations',
                    'algèbre linéaire': 'Linear Algebra',
                    'analyse': 'Analysis',
                    'topologie': 'Topology',
                    'théorie des nombres': 'Number Theory'
                },
                'Chemistry': {
                    'chimie': 'Chemistry',
                    'chimie organique': 'Organic Chemistry',
                    'chimie inorganique': 'Inorganic Chemistry',
                    'chimie physique': 'Physical Chemistry',
                    'biochimie': 'Biochemistry',
                    'chimie analytique': 'Analytical Chemistry',
                    'thermodynamique chimique': 'Chemical Thermodynamics',
                    'cinétique chimique': 'Chemical Kinetics',
                    'équilibre chimique': 'Chemical Equilibrium',
                    'liaisons chimiques': 'Chemical Bonds',
                    'réactions chimiques': 'Chemical Reactions'
                },
                'Biology': {
                    'biologie': 'Biology',
                    'biologie moléculaire': 'Molecular Biology',
                    'biologie cellulaire': 'Cell Biology',
                    'génétique': 'Genetics',
                    'évolution': 'Evolution',
                    'écologie': 'Ecology',
                    'physiologie': 'Physiology',
                    'anatomie': 'Anatomy',
                    'microbiologie': 'Microbiology',
                    'neurosciences': 'Neuroscience',
                    'biologie du développement': 'Developmental Biology',
                    'biotechnologie': 'Biotechnology'
                }
            },
            'german': {
                'Physics': {
                    'physik': 'Physics',
                    'mechanik': 'Mechanics',
                    'klassische mechanik': 'Classical Mechanics',
                    'quantenmechanik': 'Quantum Mechanics',
                    'thermodynamik': 'Thermodynamics',
                    'elektromagnetismus': 'Electromagnetism',
                    'optik': 'Optics',
                    'relativitätstheorie': 'Relativity',
                    'energie': 'Energy',
                    'kraft': 'Force',
                    'geschwindigkeit': 'Velocity',
                    'beschleunigung': 'Acceleration',
                    'impuls': 'Momentum',
                    'elektrisches feld': 'Electric Field',
                    'magnetisches feld': 'Magnetic Field',
                    'wellen': 'Waves',
                    'schwingungen': 'Oscillations',
                    'kernphysik': 'Nuclear Physics',
                    'atomphysik': 'Atomic Physics',
                    'astrophysik': 'Astrophysics',
                    'kondensierte materie': 'Condensed Matter'
                },
                'Mathematics': {
                    'mathematik': 'Mathematics',
                    'analysis': 'Calculus',
                    'algebra': 'Algebra',
                    'geometrie': 'Geometry',
                    'trigonometrie': 'Trigonometry',
                    'statistik': 'Statistics',
                    'wahrscheinlichkeitsrechnung': 'Probability',
                    'differentialgleichungen': 'Differential Equations',
                    'lineare algebra': 'Linear Algebra',
                    'analysis': 'Analysis',
                    'topologie': 'Topology',
                    'zahlentheorie': 'Number Theory'
                },
                'Chemistry': {
                    'chemie': 'Chemistry',
                    'organische chemie': 'Organic Chemistry',
                    'anorganische chemie': 'Inorganic Chemistry',
                    'physikalische chemie': 'Physical Chemistry',
                    'biochemie': 'Biochemistry',
                    'analytische chemie': 'Analytical Chemistry',
                    'chemische thermodynamik': 'Chemical Thermodynamics',
                    'chemische kinetik': 'Chemical Kinetics',
                    'chemisches gleichgewicht': 'Chemical Equilibrium',
                    'chemische bindungen': 'Chemical Bonds',
                    'chemische reaktionen': 'Chemical Reactions'
                },
                'Biology': {
                    'biologie': 'Biology',
                    'molekularbiologie': 'Molecular Biology',
                    'zellbiologie': 'Cell Biology',
                    'genetik': 'Genetics',
                    'evolution': 'Evolution',
                    'ökologie': 'Ecology',
                    'physiologie': 'Physiology',
                    'anatomie': 'Anatomy',
                    'mikrobiologie': 'Microbiology',
                    'neurowissenschaften': 'Neuroscience',
                    'entwicklungsbiologie': 'Developmental Biology',
                    'biotechnologie': 'Biotechnology'
                }
            },
            'italian': {
                'Physics': {
                    'fisica': 'Physics',
                    'meccanica': 'Mechanics',
                    'meccanica classica': 'Classical Mechanics',
                    'meccanica quantistica': 'Quantum Mechanics',
                    'termodinamica': 'Thermodynamics',
                    'elettromagnetismo': 'Electromagnetism',
                    'ottica': 'Optics',
                    'relatività': 'Relativity',
                    'energia': 'Energy',
                    'forza': 'Force',
                    'velocità': 'Velocity',
                    'accelerazione': 'Acceleration',
                    'momento': 'Momentum',
                    'campo elettrico': 'Electric Field',
                    'campo magnetico': 'Magnetic Field',
                    'onde': 'Waves',
                    'oscillazioni': 'Oscillations',
                    'fisica nucleare': 'Nuclear Physics',
                    'fisica atomica': 'Atomic Physics',
                    'astrofisica': 'Astrophysics',
                    'materia condensata': 'Condensed Matter'
                }
            },
            'polish': {
                'Physics': {
                    # Core Physics Terms
                    'fizyka': 'Physics',
                    'fizyka dla szkół wyższych': 'University Physics',
                    'physics dla szkół wyższych': 'University Physics',
                    'mechanika': 'Mechanics',
                    'mechanika klasyczna': 'Classical Mechanics',
                    'mechanika kwantowa': 'Quantum Mechanics',
                    'termodynamika': 'Thermodynamics',
                    'elektromagnetyzm': 'Electromagnetism',
                    'optyka': 'Optics',
                    'teoria względności': 'Relativity',
                    'energia': 'Energy',
                    'siła': 'Force',
                    'prędkość': 'Velocity',
                    'velocity': 'Velocity',
                    'przyspieszenie': 'Acceleration',
                    'pęd': 'Momentum',
                    'momentum': 'Momentum',
                    'pole elektryczne': 'Electric Field',
                    'electric field': 'Electric Field',
                    'pole magnetyczne': 'Magnetic Field',
                    'magnetic field': 'Magnetic Field',
                    'fale': 'Waves',
                    'waves': 'Waves',
                    'oscylacje': 'Oscillations',
                    'oscillations': 'Oscillations',
                    'fizyka jądrowa': 'Nuclear Physics',
                    'fizyka atomowa': 'Atomic Physics',
                    'astrofizyka': 'Astrophysics',
                    'materia skondensowana': 'Condensed Matter',
                    
                    # Thermodynamics terms
                    'wstęp': 'Introduction',
                    'temperatura i równowaga termiczna': 'Temperature and Thermal Equilibrium',
                    'termometry i skale temperatur': 'Thermometry and Temperature Scales',
                    'rozszerzalność cieplna': 'Thermal Expansion',
                    'przekazywanie ciepła, ciepło właściwe i kalorymetria': 'Heat Transfer, Specific Heat, and Calorimetry',
                    'przemiany fazowe': 'Phase Transitions',
                    'mechanizmy przekazywania ciepła': 'Heat Transfer Mechanisms',
                    'model cząsteczkowy gazu doskonałego': 'Molecular Model of an Ideal Gas',
                    'ciśnienie, temperatura i średnia': 'Pressure, Temperature, and Average',
                    'velocity kwadratowa cząsteczek': 'Velocity Squared of Molecules',
                    'ciepło właściwe i zasada ekwipartycji energii': 'Specific Heat and Equipartition of Energy',
                    'rozkład prędkości cząsteczek gazu doskonałego': 'Distribution of Molecular Velocities in an Ideal Gas',
                    'układy termodynamiczne': 'Thermodynamic Systems',
                    'praca, ciepło i': 'Work, Heat, and',
                    'energy wewnętrzna': 'Internal Energy',
                    'pierwsza zasada termodynamiki': 'First Law of Thermodynamics',
                    'procesy termodynamiczne': 'Thermodynamic Processes',
                    'pojemność cieplna gazu doskonałego': 'Heat Capacity of an Ideal Gas',
                    'proces adiabatyczny gazu doskonałego': 'Adiabatic Process of an Ideal Gas',
                    'procesy odwracalne i nieodwracalne': 'Reversible and Irreversible Processes',
                    'silniki cieplne': 'Heat Engines',
                    'chłodziarki i pompy ciepła': 'Refrigerators and Heat Pumps',
                    'sformułowania drugiej zasady termodynamiki': 'Statements of the Second Law of Thermodynamics',
                    'cykl carnota': 'Carnot Cycle',
                    'entropia': 'Entropy',
                    'entropia w skali mikroskopowej': 'Entropy on a Microscopic Scale',
                    'druga zasada termodynamiki': 'Second Law of Thermodynamics',
                    
                    # Heat and Temperature
                    'temperatura i ciepło': 'Temperature and Heat',
                    
                    # Kinetic Theory
                    'kinetyczna': 'Kinetic',
                    'theory gazów': 'Theory of Gases',
                    
                    # Electricity and Magnetism
                    'elektryczność i magnetyzm': 'Electricity and Magnetism',
                    'ładunek elektryczny': 'Electric Charge',
                    'przewodniki, izolatory i elektryzowanie przez indukcję': 'Conductors, Insulators, and Charging by Induction',
                    'prawo coulomba': 'Coulomb\'s Law',
                    'wyznaczanie natężenia pola elektrycznego rozkładu ładunków': 'Calculating Electric Field of a Charge Distribution',
                    'linie pola elektrycznego': 'Electric Field Lines',
                    'dipole elektryczne': 'Electric Dipoles',
                    'strumień pola elektrycznego': 'Electric Flux',
                    'wyjaśnienie prawa gaussa': 'Gauss\'s Law',
                    'stosowanie prawa gaussa': 'Applications of Gauss\'s Law',
                    'przewodniki w stanie równowagi elektrostatycznej': 'Conductors in Electrostatic Equilibrium',
                    'elektryczna': 'Electric',
                    'energy potencjalna': 'Potential Energy',
                    'potencjał elektryczny i różnica potencjałów': 'Electric Potential and Potential Difference',
                    'obliczanie potencjału elektrycznego': 'Calculating Electric Potential',
                    'obliczanie natężenia na podstawie potencjału': 'Calculating Electric Field from Potential',
                    'powierzchnie ekwipotencjalne i przewodniki': 'Equipotential Surfaces and Conductors',
                    'zastosowanie elektrostatyki': 'Applications of Electrostatics',
                    'kondensatory i pojemność elektryczna': 'Capacitors and Capacitance',
                    'łączenie szeregowe i równoległe kondensatorów': 'Series and Parallel Combinations of Capacitors',
                    'energy zgromadzona w kondensatorze': 'Energy Stored in a Capacitor',
                    'kondensator z dielektrykiem': 'Capacitor with a Dielectric',
                    'mikroskopowy model dielektryka': 'Microscopic Model of Dielectrics',
                    'prąd elektryczny': 'Electric Current',
                    'model przewodnictwa w metalach': 'Model for Electrical Conduction in Metals',
                    'rezystywność i rezystancja': 'Resistivity and Resistance',
                    'prawo ohma': "Ohm's Law",
                    'energy i moc elektryczna': 'Electric Energy and Power',
                    'nadprzewodniki': 'Superconductors',
                    'force elektromotoryczna': 'Electromotive Force',
                    'oporniki połączone szeregowo i równolegle': 'Resistors in Series and Parallel',
                    'prawa kirchhoffa': "Kirchhoff's Laws",
                    'elektryczne przyrządy pomiarowe': 'Electrical Measuring Instruments',
                    'obwody rc': 'RC Circuits',
                    'instalacja elektryczna w domu i bezpieczeństwo elektryczne': 'Household Electrical Wiring and Safety',
                    'odkrywanie magnetyzmu': 'Discovering Magnetism',
                    'pola magnetyczne i ich linie': 'Magnetic Fields and Field Lines',
                    'ruch cząstki naładowanej w polu magnetycznym': 'Motion of a Charged Particle in a Magnetic Field',
                    'force magnetyczna działająca na przewodnik z prądem': 'Magnetic Force on a Current-Carrying Conductor',
                    'wypadkowa sił i moment sił działających na pętlę z prądem': 'Force and Torque on a Current Loop',
                    'efekt halla': 'The Hall Effect',
                    'zastosowania sił i pól magnetycznych': 'Applications of Magnetic Forces and Fields',
                    'prawo biota-savarta': 'The Biot-Savart Law',
                    'cienkiego, prostoliniowego przewodu z prądem': 'Magnetic Field of a Thin Straight Wire',
                    'oddziaływanie magnetyczne dwóch równoległych przewodów z prądem': 'Magnetic Force between Two Parallel Conductors',
                    'pętli z prądem': 'Magnetic Field of a Current Loop',
                    'prawo ampère\'a': "Ampère's Law",
                    'solenoidy i toroidy': 'Solenoids and Toroids',
                    'magnetyzm materii': 'Magnetism in Matter',
                    'prawo faradaya': "Faraday's Law",
                    'reguła lenza': "Lenz's Law",
                    'force elektromotoryczna wywołana ruchem': 'Motional Electromotive Force',
                    'indukowane pola elektryczne': 'Induced Electric Fields',
                    'prądy wirowe': 'Eddy Currents',
                    'generatory elektryczne i force przeciwelektromotoryczna': 'Electric Generators and Back EMF',
                    'zastosowania indukcji elektromagnetycznej': 'Applications of Electromagnetic Induction',
                    'indukcyjność wzajemna': 'Mutual Inductance',
                    'samoindukcja i cewki indukcyjne': 'Self-Inductance and Inductors',
                    'energy magazynowana w polu magnetycznym': 'Energy Stored in a Magnetic Field',
                    'obwody rl': 'RL Circuits',
                    'obwodów lc': 'LC Oscillations',
                    'obwody rlc': 'RLC Circuits',
                    'źródła prądu zmiennego': 'AC Sources',
                    'proste obwody prądu zmiennego': 'Simple AC Circuits',
                    'obwody szeregowe rlc prądu zmiennego': 'RLC Series AC Circuits',
                    'moc w obwodzie prądu zmiennego': 'Power in an AC Circuit',
                    'rezonans w obwodzie prądu zmiennego': 'Resonance in AC Circuits',
                    'transformatory': 'Transformers',
                    'równania maxwella i': "Maxwell's Equations and",
                    'elektromagnetyczne': 'Electromagnetic',
                    'płaskie': 'Plane',
                    'niesiona przez': 'Carried by',
                    'ciśnienie promieniowania elektromagnetycznego': 'Radiation Pressure',
                    'widmo promieniowania elektromagnetycznego': 'The Electromagnetic Spectrum',
                    
                    # General terms
                    'ładunki i pola elektryczne': 'Electric Charges and Fields',
                    'prawo gaussa': "Gauss's Law",
                    'potencjał elektryczny': 'Electric Potential',
                    'pojemność elektryczna': 'Capacitance',
                    'prąd i rezystancja': 'Current and Resistance',
                    'obwody prądu stałego': 'Direct-Current Circuits',
                    'force i': 'Forces and',
                    'źródła pola magnetycznego': 'Sources of the Magnetic Field',
                    'indukcja elektromagnetyczna': 'Electromagnetic Induction',
                    'indukcyjność': 'Inductance',
                    'obwody prądu zmiennego': 'Alternating-Current Circuits',
                    
                    # Book title translations
                    'tom': 'Volume',
                    'dla szkół wyższych': 'for Universities',
                    'tom 1': 'Volume 1',
                    'tom 2': 'Volume 2',
                    'tom 3': 'Volume 3'
                }
            },
            'portuguese': {
                'Physics': {
                    'física': 'Physics',
                    'mecânica': 'Mechanics',
                    'mecânica clássica': 'Classical Mechanics',
                    'mecânica quântica': 'Quantum Mechanics',
                    'termodinâmica': 'Thermodynamics',
                    'eletromagnetismo': 'Electromagnetism',
                    'óptica': 'Optics',
                    'relatividade': 'Relativity',
                    'energia': 'Energy',
                    'força': 'Force',
                    'velocidade': 'Velocity',
                    'aceleração': 'Acceleration',
                    'momento': 'Momentum',
                    'campo elétrico': 'Electric Field',
                    'campo magnético': 'Magnetic Field',
                    'ondas': 'Waves',
                    'oscilações': 'Oscillations',
                    'física nuclear': 'Nuclear Physics',
                    'física atômica': 'Atomic Physics',
                    'astrofísica': 'Astrophysics',
                    'matéria condensada': 'Condensed Matter'
                }
            }
        }
    
    def _initialize_common_academic_terms(self) -> Dict[str, Dict[str, str]]:
        """
        Goal: Initialize common academic terms across languages.
        
        Structure: {language: {foreign_term: english_term}}
        """
        return {
            'spanish': {
                'capítulo': 'Chapter',
                'sección': 'Section',
                'introducción': 'Introduction',
                'conclusión': 'Conclusion',
                'resumen': 'Summary',
                'ejercicios': 'Exercises',
                'problemas': 'Problems',
                'teoría': 'Theory',
                'práctica': 'Practice',
                'laboratorio': 'Laboratory',
                'experimento': 'Experiment',
                'método': 'Method',
                'análisis': 'Analysis',
                'resultados': 'Results',
                'discusión': 'Discussion',
                'referencias': 'References',
                'bibliografía': 'Bibliography'
            },
            'french': {
                'chapitre': 'Chapter',
                'section': 'Section',
                'introduction': 'Introduction',
                'conclusion': 'Conclusion',
                'résumé': 'Summary',
                'exercices': 'Exercises',
                'problèmes': 'Problems',
                'théorie': 'Theory',
                'pratique': 'Practice',
                'laboratoire': 'Laboratory',
                'expérience': 'Experiment',
                'méthode': 'Method',
                'analyse': 'Analysis',
                'résultats': 'Results',
                'discussion': 'Discussion',
                'références': 'References',
                'bibliographie': 'Bibliography'
            },
            'german': {
                'kapitel': 'Chapter',
                'abschnitt': 'Section',
                'einführung': 'Introduction',
                'schlussfolgerung': 'Conclusion',
                'zusammenfassung': 'Summary',
                'übungen': 'Exercises',
                'probleme': 'Problems',
                'theorie': 'Theory',
                'praxis': 'Practice',
                'labor': 'Laboratory',
                'experiment': 'Experiment',
                'methode': 'Method',
                'analyse': 'Analysis',
                'ergebnisse': 'Results',
                'diskussion': 'Discussion',
                'referenzen': 'References',
                'literatur': 'Bibliography'
            },
            'italian': {
                'capitolo': 'Chapter',
                'sezione': 'Section',
                'introduzione': 'Introduction',
                'conclusione': 'Conclusion',
                'riassunto': 'Summary',
                'esercizi': 'Exercises',
                'problemi': 'Problems',
                'teoria': 'Theory',
                'pratica': 'Practice',
                'laboratorio': 'Laboratory',
                'esperimento': 'Experiment',
                'metodo': 'Method',
                'analisi': 'Analysis',
                'risultati': 'Results',
                'discussione': 'Discussion',
                'riferimenti': 'References',
                'bibliografia': 'Bibliography'
            },
            'polish': {
                'rozdział': 'Chapter',
                'sekcja': 'Section',
                'wprowadzenie': 'Introduction',
                'wniosek': 'Conclusion',
                'podsumowanie': 'Summary',
                'ćwiczenia': 'Exercises',
                'problemy': 'Problems',
                'teoria': 'Theory',
                'praktyka': 'Practice',
                'laboratorium': 'Laboratory',
                'eksperyment': 'Experiment',
                'metoda': 'Method',
                'analiza': 'Analysis',
                'wyniki': 'Results',
                'dyskusja': 'Discussion',
                'referencje': 'References',
                'bibliografia': 'Bibliography'
            },
            'portuguese': {
                'capítulo': 'Chapter',
                'seção': 'Section',
                'introdução': 'Introduction',
                'conclusão': 'Conclusion',
                'resumo': 'Summary',
                'exercícios': 'Exercises',
                'problemas': 'Problems',
                'teoria': 'Theory',
                'prática': 'Practice',
                'laboratório': 'Laboratory',
                'experimento': 'Experiment',
                'método': 'Method',
                'análise': 'Analysis',
                'resultados': 'Results',
                'discussão': 'Discussion',
                'referências': 'References',
                'bibliografia': 'Bibliography'
            }
        }
    
    def _get_translations_for_language(self, language: str, discipline: str) -> Dict[str, str]:
        """
        Goal: Get combined translation mappings for a language and discipline.
        """
        translations = {}
        
        # Add common academic terms
        if language in self.common_academic_terms:
            translations.update(self.common_academic_terms[language])
        
        # Add discipline-specific terms
        if language in self.discipline_translations:
            if discipline in self.discipline_translations[language]:
                translations.update(self.discipline_translations[language][discipline])
        
        return translations
    
    def _apply_translations(self, text: str, translations: Dict[str, str]) -> str:
        """
        Goal: Apply translation mappings to text.
        """
        result = text.lower()
        
        # Sort by length (longest first) to handle overlapping terms
        sorted_terms = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)
        
        for foreign_term, english_term in sorted_terms:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(foreign_term.lower()) + r'\b'
            result = re.sub(pattern, english_term, result, flags=re.IGNORECASE)
        
        return result.title()  # Convert to title case
    
    def _standardize_english_terminology(self, text: str, discipline: str) -> str:
        """
        Goal: Standardize English terminology for the discipline.
        """
        # Standard terminology mappings for English
        standardizations = {
            'Physics': {
                'newtons law': "Newton's Law",
                'newtons laws': "Newton's Laws",
                'maxwells equations': "Maxwell's Equations",
                'schrodingers equation': "Schrödinger's Equation",
                'force and motion': "Forces and Motion",
                'work and energy': "Work, Energy, and Power",
                'electromagnetic field': "Electromagnetic Fields",
                'quantum mechanic': "Quantum Mechanics",
                'classical mechanic': "Classical Mechanics",
                'thermal physic': "Thermal Physics",
                'nuclear physic': "Nuclear Physics",
                'atomic physic': "Atomic Physics",
                'particle physic': "Particle Physics",
                'solid state': "Condensed Matter",
                'astro physics': "Astrophysics"
            },
            'Mathematics': {
                'calculus': "Calculus",
                'linear algebra': "Linear Algebra",
                'differential equation': "Differential Equations",
                'number theory': "Number Theory",
                'abstract algebra': "Abstract Algebra",
                'real analysis': "Real Analysis",
                'complex analysis': "Complex Analysis",
                'topology': "Topology",
                'probability theory': "Probability Theory",
                'mathematical statistics': "Mathematical Statistics"
            },
            'Chemistry': {
                'organic chemistry': "Organic Chemistry",
                'inorganic chemistry': "Inorganic Chemistry",
                'physical chemistry': "Physical Chemistry",
                'analytical chemistry': "Analytical Chemistry",
                'biochemistry': "Biochemistry",
                'chemical thermodynamics': "Chemical Thermodynamics",
                'chemical kinetics': "Chemical Kinetics",
                'quantum chemistry': "Quantum Chemistry"
            },
            'Biology': {
                'molecular biology': "Molecular Biology",
                'cell biology': "Cell Biology",
                'developmental biology': "Developmental Biology",
                'evolutionary biology': "Evolutionary Biology",
                'marine biology': "Marine Biology",
                'plant biology': "Plant Biology",
                'animal biology': "Animal Biology",
                'computational biology': "Computational Biology"
            }
        }
        
        if discipline in standardizations:
            text_lower = text.lower()
            for key, standard in standardizations[discipline].items():
                if key in text_lower:
                    return standard
        
        # Default: clean up and title case
        return text.strip().title()
    
    def get_supported_languages(self) -> List[str]:
        """
        Goal: Get list of supported languages for translation.
        """
        return list(self.discipline_translations.keys())
    
    def get_supported_disciplines(self, language: str) -> List[str]:
        """
        Goal: Get list of supported disciplines for a language.
        """
        if language in self.discipline_translations:
            return list(self.discipline_translations[language].keys())
        return []
    
    def translate_book_title(self, title: str, source_language: str) -> str:
        """
        Goal: Translate book titles to English with proper formatting.
        
        Args:
            title: Original book title
            source_language: Source language of the title
            
        Returns:
            Translated book title in English
        """
        if source_language.lower() == 'english':
            return title
        
        # Handle special book title patterns
        book_title_patterns = {
            'polish': {
                'fizyka-dla-szkół-wyższych-tom-1': 'University Physics Volume 1',
                'fizyka-dla-szkół-wyższych-tom-2': 'University Physics Volume 2', 
                'fizyka-dla-szkół-wyższych-tom-3': 'University Physics Volume 3',
                'fizyka-dla-szkół-wyższych-tom-2.collection': 'University Physics Volume 2',
                'psychologia': 'Psychology',
                'psychologia.collection': 'Psychology',
                'mikroekonomia-podstawy': 'Microeconomics Fundamentals',
                'mikroekonomia-podstawy.collection': 'Microeconomics Fundamentals'
            },
            'spanish': {
                'física-universitaria-volumen-1': 'University Physics Volume 1',
                'física-universitaria-volumen-2': 'University Physics Volume 2',
                'física-universitaria-volumen-3': 'University Physics Volume 3',
                'física-universitaria-volumen-2.collection': 'University Physics Volume 2',
                'química-2ed': 'Chemistry 2nd Edition',
                'química-2ed.collection': 'Chemistry 2nd Edition',
                'introducción-estadística': 'Introduction to Statistics',
                'introducción-estadística.collection': 'Introduction to Statistics',
                'prealgebra-2e': 'Prealgebra 2nd Edition',
                'prealgebra-2e.collection': 'Prealgebra 2nd Edition'
            },
            'french': {
                'introduction-anthropology': 'Introduction to Anthropology',
                'introduction-anthropology.collection': 'Introduction to Anthropology',
                'introduction-philosophy': 'Introduction to Philosophy',
                'introduction-philosophy.collection': 'Introduction to Philosophy',
                'introduction-political-science': 'Introduction to Political Science',
                'introduction-political-science.collection': 'Introduction to Political Science',
                'introduction-sociology-3e': 'Introduction to Sociology 3rd Edition',
                'introduction-sociology-3e.collection': 'Introduction to Sociology 3rd Edition',
                'introduction-python-programming': 'Introduction to Python Programming',
                'introduction-python-programming.collection': 'Introduction to Python Programming',
                'introduction-intellectual-property': 'Introduction to Intellectual Property',
                'introduction-intellectual-property.collection': 'Introduction to Intellectual Property',
                'introduction-business': 'Introduction to Business',
                'introduction-business.collection': 'Introduction to Business'
            },
            'german': {
                'organizational-behavior': 'Organizational Behavior',
                'organizational-behavior.collection': 'Organizational Behavior'
            }
        }
        
        # Check for exact matches first
        if source_language in book_title_patterns:
            if title in book_title_patterns[source_language]:
                return book_title_patterns[source_language][title]
        
        # Apply general translation rules
        translated = self.translate_to_english(title, source_language, 'Physics')
        
        # Clean up common title formatting
        translated = translated.replace('.Collection', '')
        translated = translated.replace('.collection', '')
        translated = translated.replace('-', ' ')
        translated = re.sub(r'\s+', ' ', translated)  # Remove extra spaces
        
        return translated.strip().title()
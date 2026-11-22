import './style.css';
import shell from './shell.html?raw';

// Keep the Voice Agent page and native audio lifecycle together; React does not own these nodes.
document.getElementById('root').innerHTML = shell;
void import('./ksa.js');

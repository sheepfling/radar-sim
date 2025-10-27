import {describeObject, loadMeshFromFile} from './meshLoader';
import {MeshViewer} from './viewer';

const container = document.getElementById('viewer')!;
const viewer = new MeshViewer(container);

const input = document.getElementById('fileInput') as HTMLInputElement;
const button = document.getElementById('loadButton') as HTMLButtonElement;
const infoBox = document.getElementById('infoBox') as HTMLTextAreaElement;

button.addEventListener('click', async () => {
    const file = input.files?.[0];
    if (!file) {
        return alert('Please select a file first.');
    }

    viewer.clear();

    const obj = await loadMeshFromFile(file);
    viewer.loadObject(obj);
    infoBox.value = describeObject(obj);
});
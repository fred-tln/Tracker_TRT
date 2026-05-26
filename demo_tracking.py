#!/usr/bin/env python3
"""
Demo interactive pour OSTrack - Pistoage visuel sur vidéo
Permet de sélectionner une vidéo, de désigner une cible avec la souris,
puis de lancer le pistage en temps réel.

Utilisation:
    python demo_tracking.py

Contrôles:
    - Sélectionnez la zone d'intérêt avec la souris, puis appuyez sur Entrée
    - Appuyez sur 'q' pour quitter
    - Appuyez sur 'r' pour réinitialiser (choisir une nouvelle cible)
"""

import os
import sys
import cv2
import torch
import argparse
from pathlib import Path

# Ajoute le répertoire racine du projet au path
prj_path = os.path.dirname(os.path.abspath(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation.tracker import Tracker
from lib.test.utils.hann import hann2d


def select_video():
    """Ouvre une boîte de dialogue pour sélectionner un fichier vidéo."""
    # Utilisation de la méthode OpenCV pour ouvrir un fichier
    cv2.namedWindow("Sélection de la vidéo", cv2.WINDOW_NORMAL)
    
    # Sur Linux, on peut utiliser zenity ou kdialog
    # Sur Windows, on utilise filedialog de tkinter
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()  # Cache la fenêtre principale
        root.attributes('-topmost', True)  # Met la fenêtre au premier plan
        
        video_path = filedialog.askopenfilename(
            title="Sélectionnez une vidéo",
            filetypes=[
                ("Fichiers vidéo", "*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.webm"),
                ("Tous les fichiers", "*.*")
            ]
        )
        root.destroy()
        
        if not video_path:
            return None
        return video_path
        
    except ImportError:
        # Fallback: saisie manuelle du chemin
        print("\nImpossible d'ouvrir une boîte de dialogue.")
        print("Veuillez saisir le chemin de la vidéo:")
        video_path = input("> ").strip()
        if os.path.isfile(video_path):
            return video_path
        return None


def load_tracker(tracker_name='ostrack', param_name='vitb_256_mae_ce_32x4_ep300'):
    """Charge le tracker OSTrack avec les paramètres par défaut."""
    print(f"\nChargement du tracker {tracker_name}...")
    
    tracker = Tracker(tracker_name, param_name, 'demo')
    params = tracker.get_parameters()
    
    # Désactive le mode debug pour la démo
    params.debug = 0
    params.save_all_boxes = False
    
    tracker_instance = tracker.create_tracker(params)
    print("Tracker chargé avec succès !")
    
    return tracker_instance


def select_roi(frame, window_name="Sélection de la cible"):
    """Permet à l'utilisateur de sélectionner une région d'intérêt avec la souris."""
    # Affiche un message sur l'image
    frame_disp = frame.copy()
    cv2.putText(frame_disp, 'Sélectionnez la cible avec la souris, puis appuyez sur Entrée', 
                (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 255, 0), 2)
    cv2.putText(frame_disp, 'Cliquez et glissez pour dessiner la zone', 
                (20, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 200, 200), 1)
    
    cv2.imshow(window_name, frame_disp)
    
    # Utilise selectROI d'OpenCV (mode free=False pour un rectangle)
    # Le mode fromCenter=False permet de dessiner depuis un coin
    roi = cv2.selectROI(window_name, frame_disp, fromCenter=False, showCrosshair=True)
    
    # Supprime la fenêtre de sélection
    cv2.destroyWindow(f"{window_name}-ROI")
    
    return list(roi)


def run_tracking(video_path, tracker, window_name="OSTrack"):
    """
    Lance le pistage sur la vidéo.
    
    Args:
        video_path: Chemin vers la vidéo
        tracker: Instance du tracker OSTrack
        window_name: Nom de la fenêtre d'affichage
    
    Returns:
        Liste des boîtes de décision prédites
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo '{video_path}'")
        return []
    
    # Récupère les informations de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nInformation vidéo:")
    print(f"  Résolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"\nContrôles:")
    print(f"  - 'q': Quitter")
    print(f"  - 'r': Réinitialiser (nouvelle cible)")
    print(f"  - ' ': Pause/Dépause")
    print(f"  - 's': Sauvegarder la vidéo avec les résultats")
    print()
    
    # Crée la fenêtre d'affichage
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, min(960, width), min(720, height))
    
    output_boxes = []
    paused = False
    frame_idx = 0
    
    # Enregistreur pour sauvegarde optionnelle
    writer = None
    output_video_path = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nFin de la vidéo atteinte.")
            break
        
        # Mode pause
        if paused:
            cv2.putText(frame, 'PAUSE (appuyez sur " " pour continuer)', 
                        (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 255, 255), 2)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(' '):
                paused = False
            elif key == ord('q'):
                break
            elif key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rembobine
                break
            continue
        
        # Frame 0: Initialisation
        if frame_idx == 0:
            print("Sélectionnez la cible à pister...")
            init_bbox = select_roi(frame, window_name)
            
            if init_bbox is None:
                print("Aucune cible sélectionnée. Quitte.")
                break
            
            print(f"Cible sélectionnée: x={init_bbox[0]}, y={init_bbox[1]}, "
                  f"w={init_bbox[2]}, h={init_bbox[3]}")
            
            # Initialise le tracker
            tracker.initialize(frame, {'init_bbox': init_bbox})
            output_boxes.append(init_bbox)
            frame_idx += 1
            continue
        
        # Frames suivantes: Pistoage
        import time
        start_time = time.time()
        
        # Exécute le tracking
        out = tracker.track(frame)
        state = out['target_bbox']
        
        exec_time = time.time() - start_time
        output_boxes.append(state)
        
        # Convertit en entiers pour le dessin
        x, y, w, h = [int(s) for s in state]
        
        # Dessine la boîte de décision
        frame_disp = frame.copy()
        cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Ajoute un label avec les coordonnées
        label = f"x:{x} y:{y} w:{w} h:{h}"
        cv2.putText(frame_disp, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 2)
        
        # Affiche les informations en haut à gauche
        cv2.putText(frame_disp, f'Frame: {frame_idx}/{total_frames}', 
                    (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 255), 2)
        cv2.putText(frame_disp, f'Time: {exec_time*1000:.1f}ms ({1/exec_time:.1f} FPS)', 
                    (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 255), 2)
        
        # Affiche
        cv2.imshow(window_name, frame_disp)
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            print("\nRéinitialisation...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            if writer is None:
                output_video_path = str(Path(video_path).with_suffix('_tracked.mp4'))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                print(f"\nSauvegarde de la vidéo: {output_video_path}")
            writer.write(frame_disp)
        
        frame_idx += 1
    
    # Nettoyage
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Vidéo sauvegardée: {output_video_path}")
    
    cv2.destroyWindow(window_name)
    
    return output_boxes


def main():
    parser = argparse.ArgumentParser(description='OSTrack Demo Interactive')
    parser.add_argument('--tracker', type=str, default='ostrack',
                        help='Nom du tracker (default: ostrack)')
    parser.add_argument('--param', type=str, default='vitb_256_mae_ce_32x4_ep300',
                        help='Nom du paramètre du tracker')
    parser.add_argument('--video', type=str, default=None,
                        help='Chemin vers la vidéo (sinon boîte de dialogue)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  OSTrack - Demo Interactive de Pistoage Visuel")
    print("=" * 60)
    
    # Vérifie que CUDA est disponible
    if torch.cuda.is_available():
        print(f"\nCUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("\nAttention: CUDA non disponible. Le tracking sera plus lent.")
    
    # Sélection de la vidéo
    if args.video:
        video_path = args.video
        if not os.path.isfile(video_path):
            print(f"Erreur: Le fichier '{video_path}' n'existe pas.")
            sys.exit(1)
    else:
        video_path = select_video()
        if video_path is None:
            print("Aucune vidéo sélectionnée. Quitte.")
            sys.exit(0)
    
    print(f"\nVidéo sélectionnée: {video_path}")
    
    # Charge le tracker
    try:
        tracker = load_tracker(args.tracker, args.param)
    except Exception as e:
        print(f"\nErreur lors du chargement du tracker: {e}")
        print("\nAssurez-vous que le checkpoint du modèle est présent.")
        print(f"Vérifiez le répertoire: {os.path.join(prj_path, 'pretrained_models')}")
        sys.exit(1)
    
    # Boucle principale pour permettre plusieurs essais
    while True:
        output_boxes = run_tracking(video_path, tracker)
        
        if not output_boxes:
            break
        
        print(f"\nPistoage terminé. {len(output_boxes)} boîtes prédites.")
        
        # Demande si l'utilisateur veut réessayer
        response = input("\nVoulez-vous réessayer avec une nouvelle cible? (o/n): ").strip().lower()
        if response != 'o':
            break
    
    print("\nAu revoir !")


if __name__ == '__main__':
    main()

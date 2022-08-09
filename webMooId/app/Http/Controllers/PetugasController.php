<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class PetugasController extends Controller
{
    public function index()
    {
        $data = array(
            'title' => 'Petugas'
        );
        return view('peternak.pages.petugas.index')->with($data);
    }

    public function create()
    {
        $data = array(
            'title' => 'Tambah Petugas Baru'
        );
        return view('peternak.pages.petugas.form')->with($data);
    }

    public function show($id)
    {
        return redirect(route('petugas.index'));
    }

    public function edit($id)
    {
        $data = array(
            'title' => 'Edit Petugas #'.$id
        );
        return view('peternak.pages.petugas.form')->with($data);
    }

    public function update(Request $request, $id)
    {
        return redirect(route('petugas.index'))->with('success', 'Petugas #'.$id.' berhasil diubah!');
    }

    public function destroy(Request $request, $id)
    {
        return redirect(route('petugas.index'))->with('success', 'Petugas #'.$id.' berhasil dihapus!');
    }
}

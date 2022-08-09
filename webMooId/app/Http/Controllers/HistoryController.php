<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class HistoryController extends Controller
{
    public function index()
    {
        $data = array(
            'title' => 'History'
        );
        return view('peternak.pages.history.index')->with($data);
    }

    public function create()
    {
        $data = array(
            'title' => 'Tambah Rekam Medis'
        );
        return view('peternak.pages.history.form')->with($data);
    }
}
